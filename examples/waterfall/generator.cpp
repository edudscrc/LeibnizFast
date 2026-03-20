/*
 * generator.cpp — DAS (Distributed Acoustic Sensing) data generator → ZeroMQ PUSH
 *
 * Simulates a DAS acquisition system streaming float32 column data over ZMQ.
 * A Python bridge (bridge.py) relays messages to WebSocket clients.
 *
 * Two ZMQ sockets:
 *   tcp://127.0.0.1:5555  PUSH — binary column data out to bridge
 *   tcp://127.0.0.1:5556  PULL — control messages in from bridge
 *
 * Control message: 4-byte little-endian uint32 = new row count.
 *
 * --- Protocol: waterfall v1 (magic 0x4C465A10, 16-byte header) -----------
 *
 *   Offset  0: magic      = 0x4C465A10
 *   Offset  4: rows       (spatial samples per column)
 *   Offset  8: new_cols   (number of columns in this message)
 *   Offset 12: msg_id     (monotonically increasing counter)
 *   Offset 16: float32[rows × new_cols]  column data (each column contiguous)
 *
 * Data model:
 *   - DAQ acquires at `sampling_rate` MHz (= MB/s of int8)
 *   - Cast int8→float32 (×4), downsample by `downsample` factor (÷N)
 *   - Effective output rate: sampling_rate × 4 / downsample MB/s
 *   - Each column = one spatial segment (rows = fiber distance samples)
 *   - Visual pattern: random noise floor + random impulse bursts
 *
 * Build:
 *   g++ -std=c++17 -O2 -o generator generator.cpp -lzmq
 *
 * Run:
 *   ./generator                                     # defaults: 100 MHz, downsample 5, 512 rows
 *   ./generator --sampling-rate 400 --downsample 5  # 320 MB/s output
 *   ./generator --rows 1024 --cols-per-msg 4        # 1024 spatial samples, 4 cols/message
 *   ./generator --debug                             # per-message timing
 */

#include <zmq.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ---- Constants -----------------------------------------------------------

static constexpr uint32_t WATERFALL_MAGIC = 0x4C465A10u;
static constexpr int HEADER_BYTES = 16;  // 4 × uint32

// Maximum number of simultaneous impulse events
static constexpr int MAX_IMPULSES = 8;
// Impulse decay rate (columns)
static constexpr int IMPULSE_DECAY_COLS = 10;

// ---- Signal handling -----------------------------------------------------

static volatile bool g_running = true;
static void handle_sigint(int) { g_running = false; }

// ---- Impulse event -------------------------------------------------------

struct Impulse {
  int row_start;    // center row of the impulse
  int row_span;     // half-width in rows
  float amplitude;  // peak amplitude
  int age;          // columns since trigger
  bool active;
};

// ---- Persistent fiber events (sine-modulated signals at fixed rows) ------

struct FiberEvent {
  int row_center;
  int row_span;
  float base_amplitude;
  float freq;  // modulation frequency (radians per column)
};

// ---- Main ----------------------------------------------------------------

int main(int argc, char* argv[]) {
  int sampling_rate_mhz = 100;
  int downsample_factor = 5;
  int rows = 512;
  int cols_per_msg = 1;
  bool debug = false;

  for (int a = 1; a < argc; ++a) {
    const std::string arg(argv[a]);
    if (arg == "--sampling-rate" && a + 1 < argc) {
      sampling_rate_mhz = std::stoi(argv[++a]);
      if (sampling_rate_mhz < 1 || sampling_rate_mhz > 10000) {
        std::cerr << "sampling-rate must be 1..10000 MHz\n";
        return 1;
      }
    } else if (arg == "--downsample" && a + 1 < argc) {
      downsample_factor = std::stoi(argv[++a]);
      if (downsample_factor < 1 || downsample_factor > 1000) {
        std::cerr << "downsample must be 1..1000\n";
        return 1;
      }
    } else if (arg == "--rows" && a + 1 < argc) {
      rows = std::stoi(argv[++a]);
      if (rows < 4 || rows > 8192) {
        std::cerr << "rows must be 4..8192\n";
        return 1;
      }
    } else if (arg == "--cols-per-msg" && a + 1 < argc) {
      cols_per_msg = std::stoi(argv[++a]);
      if (cols_per_msg < 1 || cols_per_msg > 256) {
        std::cerr << "cols-per-msg must be 1..256\n";
        return 1;
      }
    } else if (arg == "--debug") {
      debug = true;
    }
  }

  // Compute output rate
  // output_rate = sampling_rate_MHz * 4 / downsample_factor  (MB/s)
  const double output_rate_mbs =
      static_cast<double>(sampling_rate_mhz) * 4.0 / downsample_factor;
  const int bytes_per_col = rows * static_cast<int>(sizeof(float));
  const double cols_per_sec = (output_rate_mbs * 1e6) / bytes_per_col;
  const double msgs_per_sec = cols_per_sec / cols_per_msg;

  // Time between messages in nanoseconds
  const auto msg_interval_ns = static_cast<int64_t>(1e9 / msgs_per_sec);

  std::cout << "DAS Waterfall Generator\n"
            << "  Sampling rate:  " << sampling_rate_mhz << " MHz\n"
            << "  Downsample:     " << downsample_factor << "×\n"
            << "  Output rate:    " << output_rate_mbs << " MB/s\n"
            << "  Rows:           " << rows << "\n"
            << "  Cols/msg:       " << cols_per_msg << "\n"
            << "  Cols/sec:       " << cols_per_sec << "\n"
            << "  Msgs/sec:       " << msgs_per_sec << "\n"
            << "  Msg interval:   " << (msg_interval_ns / 1e6) << " ms\n"
            << std::flush;

  signal(SIGINT, handle_sigint);

  // ---- Random number generator -----------------------------------------

  std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
  std::uniform_real_distribution<float> noise_dist(-0.1f, 0.1f);
  std::uniform_real_distribution<float> amp_dist(0.5f, 1.0f);
  std::uniform_int_distribution<int> row_dist(0, rows - 1);
  std::uniform_int_distribution<int> span_dist(5, std::max(5, rows / 10));
  std::uniform_real_distribution<float> trigger_dist(0.0f, 1.0f);

  // ---- Initialize persistent fiber events ------------------------------

  const int n_fiber_events = 4;
  std::vector<FiberEvent> fiber_events(n_fiber_events);
  for (int i = 0; i < n_fiber_events; i++) {
    fiber_events[i].row_center = rows * (i + 1) / (n_fiber_events + 1);
    fiber_events[i].row_span = std::max(3, rows / 20);
    fiber_events[i].base_amplitude = 0.3f + 0.2f * (i % 2);
    fiber_events[i].freq = 0.02f + 0.01f * i;
  }

  // ---- Initialize impulse array ----------------------------------------

  std::vector<Impulse> impulses(MAX_IMPULSES);
  for (auto& imp : impulses) imp.active = false;

  // ---- ZMQ setup -------------------------------------------------------

  void* ctx = zmq_ctx_new();

  void* sock_data = zmq_socket(ctx, ZMQ_PUSH);
  {
    int sndhwm = 8;
    zmq_setsockopt(sock_data, ZMQ_SNDHWM, &sndhwm, sizeof(sndhwm));
    int linger = 0;
    zmq_setsockopt(sock_data, ZMQ_LINGER, &linger, sizeof(linger));
    if (zmq_bind(sock_data, "tcp://127.0.0.1:5555") != 0) {
      std::cerr << "zmq_bind(data) failed: " << zmq_strerror(zmq_errno())
                << "\n";
      zmq_close(sock_data);
      zmq_ctx_destroy(ctx);
      return 1;
    }
    std::cout << "ZMQ PUSH bound to tcp://127.0.0.1:5555\n";
  }

  void* sock_ctrl = zmq_socket(ctx, ZMQ_PULL);
  {
    int rcvhwm = 4;
    zmq_setsockopt(sock_ctrl, ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));
    int linger = 0;
    zmq_setsockopt(sock_ctrl, ZMQ_LINGER, &linger, sizeof(linger));
    if (zmq_bind(sock_ctrl, "tcp://127.0.0.1:5556") != 0) {
      std::cerr << "zmq_bind(ctrl) failed: " << zmq_strerror(zmq_errno())
                << "\n";
      zmq_close(sock_data);
      zmq_close(sock_ctrl);
      zmq_ctx_destroy(ctx);
      return 1;
    }
    std::cout << "ZMQ PULL bound to tcp://127.0.0.1:5556\n";
  }

  // ---- Pre-allocate send buffer ----------------------------------------

  const size_t max_payload_bytes =
      static_cast<size_t>(rows) * cols_per_msg * sizeof(float);
  std::vector<uint8_t> send_buf(HEADER_BYTES + max_payload_bytes);

  // ---- Column generation buffer ----------------------------------------

  std::vector<float> col_buf(static_cast<size_t>(rows) * cols_per_msg);

  uint32_t msg_id = 0;
  int64_t col_index = 0;

  std::cout << "Streaming... (Ctrl+C to stop)\n" << std::flush;

  auto next_send_time = std::chrono::steady_clock::now();

  while (g_running) {
    // ---- Check for control messages (non-blocking) -------------------

    uint8_t ctrl_buf[4];
    int ctrl_len = zmq_recv(sock_ctrl, ctrl_buf, sizeof(ctrl_buf), ZMQ_DONTWAIT);
    if (ctrl_len == 4) {
      uint32_t new_rows;
      memcpy(&new_rows, ctrl_buf, 4);
      if (new_rows >= 4 && new_rows <= 8192 &&
          static_cast<int>(new_rows) != rows) {
        rows = static_cast<int>(new_rows);
        std::cout << "Resize → rows=" << rows << "\n" << std::flush;

        // Reinitialize fiber events for new row count
        for (int i = 0; i < n_fiber_events; i++) {
          fiber_events[i].row_center = rows * (i + 1) / (n_fiber_events + 1);
          fiber_events[i].row_span = std::max(3, rows / 20);
        }
        row_dist = std::uniform_int_distribution<int>(0, rows - 1);
        span_dist = std::uniform_int_distribution<int>(5, std::max(5, rows / 10));

        // Reallocate buffers
        const size_t new_payload =
            static_cast<size_t>(rows) * cols_per_msg * sizeof(float);
        send_buf.resize(HEADER_BYTES + new_payload);
        col_buf.resize(static_cast<size_t>(rows) * cols_per_msg);

        // Deactivate all impulses
        for (auto& imp : impulses) imp.active = false;
      }
    }

    // ---- Generate column data ------------------------------------------

    for (int c = 0; c < cols_per_msg; c++) {
      float* col = col_buf.data() + static_cast<size_t>(c) * rows;

      // Base: noise floor
      for (int r = 0; r < rows; r++) {
        col[r] = noise_dist(rng);
      }

      // Persistent fiber events: sine-modulated Gaussian bumps
      for (const auto& ev : fiber_events) {
        float amp =
            ev.base_amplitude *
            (0.5f + 0.5f * std::sin(static_cast<float>(col_index + c) * ev.freq));
        for (int r = std::max(0, ev.row_center - ev.row_span);
             r < std::min(rows, ev.row_center + ev.row_span); r++) {
          float dist = static_cast<float>(r - ev.row_center) / ev.row_span;
          col[r] += amp * std::exp(-dist * dist * 2.0f);
        }
      }

      // Random impulse triggering (~2% chance per column)
      if (trigger_dist(rng) < 0.02f) {
        for (auto& imp : impulses) {
          if (!imp.active) {
            imp.row_start = row_dist(rng);
            imp.row_span = span_dist(rng);
            imp.amplitude = amp_dist(rng);
            imp.age = 0;
            imp.active = true;
            break;
          }
        }
      }

      // Apply active impulses
      for (auto& imp : impulses) {
        if (!imp.active) continue;
        float decay =
            1.0f -
            static_cast<float>(imp.age) / static_cast<float>(IMPULSE_DECAY_COLS);
        if (decay <= 0.0f) {
          imp.active = false;
          continue;
        }
        float a = imp.amplitude * decay;
        for (int r = std::max(0, imp.row_start - imp.row_span);
             r < std::min(rows, imp.row_start + imp.row_span); r++) {
          float dist =
              static_cast<float>(r - imp.row_start) / imp.row_span;
          col[r] += a * std::exp(-dist * dist * 2.0f);
        }
        imp.age++;
      }
    }

    col_index += cols_per_msg;

    // ---- Build message -------------------------------------------------

    const size_t payload_bytes =
        static_cast<size_t>(rows) * cols_per_msg * sizeof(float);

    // Header: magic, rows, new_cols, msg_id
    uint32_t header[4];
    header[0] = WATERFALL_MAGIC;
    header[1] = static_cast<uint32_t>(rows);
    header[2] = static_cast<uint32_t>(cols_per_msg);
    header[3] = msg_id++;

    memcpy(send_buf.data(), header, HEADER_BYTES);
    memcpy(send_buf.data() + HEADER_BYTES, col_buf.data(), payload_bytes);

    // ---- Send ----------------------------------------------------------

    const size_t total_bytes = HEADER_BYTES + payload_bytes;
    int rc = zmq_send(sock_data, send_buf.data(), total_bytes, ZMQ_DONTWAIT);

    if (debug && rc >= 0) {
      auto now = std::chrono::steady_clock::now();
      std::cout << "[perf] msg_id=" << (msg_id - 1)
                << " cols=" << cols_per_msg
                << " bytes=" << total_bytes
                << " rows=" << rows << "\n"
                << std::flush;
    }

    // ---- Throttle to target rate ---------------------------------------

    next_send_time += std::chrono::nanoseconds(msg_interval_ns);
    auto now = std::chrono::steady_clock::now();
    if (next_send_time > now) {
      std::this_thread::sleep_until(next_send_time);
    } else {
      // Falling behind — reset to avoid burst catchup
      next_send_time = now;
    }
  }

  // ---- Cleanup ---------------------------------------------------------

  std::cout << "\nShutting down...\n";
  zmq_close(sock_data);
  zmq_close(sock_ctrl);
  zmq_ctx_destroy(ctx);
  return 0;
}
