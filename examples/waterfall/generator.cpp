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
 *   Offset  4: rows       (spatial samples per column, after downsampling)
 *   Offset  8: new_cols   (number of columns in this message)
 *   Offset 12: msg_id     (monotonically increasing counter)
 *   Offset 16: float32[rows × new_cols]  column data (each column contiguous)
 *
 * Data model:
 *   - DAQ acquires int8 samples at `sampling_rate` MHz over a round-trip fiber segment
 *   - int8 cast to float32 and normalized by 128.0 → range [-0.992, 0.992]
 *   - Spatial downsampling by `spatial_downsample` (integer step)
 *   - rows = ceil(round(sampling_rate_MHz × 1e6 × 2 × fiber_length / v) / spatial_downsample)
 *   - cols_per_msg = round(repetition_rate_Hz × time_buffer_s)
 *   - msg_interval_ns = round(time_buffer_s × 1e9)
 *
 * Build:
 *   g++ -std=c++17 -O2 -o generator generator.cpp -lzmq
 *
 * Run:
 *   ./generator                                              # defaults: 400 MHz, 10–20 km fiber, ds=5
 *   ./generator --fiber-start 5000 --fiber-end 15000        # 10 km segment starting at 5 km
 *   ./generator --sampling-rate 200 --spatial-downsample 10 # lower resolution
 *   ./generator --repetition-rate 5000 --time-buffer 0.4    # slower pulse rate, larger batches
 *   ./generator --debug                                      # per-message timing
 */

#include <zmq.h>

#include <algorithm>
#include <array>
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

// Speed of light and fiber refractive index
static constexpr double C_LIGHT = 3.0e8;   // m/s
static constexpr double N_FIBER = 1.4682;  // silica single-mode fiber
static constexpr double V_FIBER = C_LIGHT / N_FIBER;  // ~2.0432e8 m/s

// ADC normalization: int8 [-127,127] → float [-0.992, 0.992]
static constexpr float ADC_NORM = 1.0f / 128.0f;

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
  double fiber_start_m      = 10000.0;  // meters
  double fiber_end_m        = 20000.0;  // meters
  int    sampling_rate_mhz  = 400;      // MHz
  int    repetition_rate_hz = 10000;    // Hz
  int    spatial_downsample = 5;        // integer step
  double time_buffer_s      = 0.2;      // seconds
  bool   debug              = false;

  for (int a = 1; a < argc; ++a) {
    const std::string arg(argv[a]);
    if (arg == "--fiber-start" && a + 1 < argc) {
      fiber_start_m = std::stod(argv[++a]);
      if (fiber_start_m < 0.0 || fiber_start_m > 1.0e6) {
        std::cerr << "fiber-start must be 0..1000000 m\n";
        return 1;
      }
    } else if (arg == "--fiber-end" && a + 1 < argc) {
      fiber_end_m = std::stod(argv[++a]);
      if (fiber_end_m < 1.0 || fiber_end_m > 1.0e6) {
        std::cerr << "fiber-end must be 1..1000000 m\n";
        return 1;
      }
    } else if (arg == "--sampling-rate" && a + 1 < argc) {
      sampling_rate_mhz = std::stoi(argv[++a]);
      if (sampling_rate_mhz < 1 || sampling_rate_mhz > 10000) {
        std::cerr << "sampling-rate must be 1..10000 MHz\n";
        return 1;
      }
    } else if (arg == "--repetition-rate" && a + 1 < argc) {
      repetition_rate_hz = std::stoi(argv[++a]);
      if (repetition_rate_hz < 1 || repetition_rate_hz > 1000000) {
        std::cerr << "repetition-rate must be 1..1000000 Hz\n";
        return 1;
      }
    } else if (arg == "--spatial-downsample" && a + 1 < argc) {
      spatial_downsample = std::stoi(argv[++a]);
      if (spatial_downsample < 1 || spatial_downsample > 1000) {
        std::cerr << "spatial-downsample must be 1..1000\n";
        return 1;
      }
    } else if (arg == "--time-buffer" && a + 1 < argc) {
      time_buffer_s = std::stod(argv[++a]);
      if (time_buffer_s < 0.001 || time_buffer_s > 60.0) {
        std::cerr << "time-buffer must be 0.001..60.0 s\n";
        return 1;
      }
    } else if (arg == "--debug") {
      debug = true;
    }
  }

  if (fiber_end_m <= fiber_start_m) {
    std::cerr << "fiber-end must be greater than fiber-start\n";
    return 1;
  }

  // ---- DAS physics computation -------------------------------------------

  const double fiber_length_m     = fiber_end_m - fiber_start_m;
  const double round_trip_time_s  = 2.0 * fiber_length_m / V_FIBER;
  const int    points_per_segment = static_cast<int>(
      std::round(sampling_rate_mhz * 1.0e6 * round_trip_time_s));
  const int    rows_init = static_cast<int>(
      std::ceil(static_cast<double>(points_per_segment) / spatial_downsample));
  const int    cols_per_msg = static_cast<int>(
      std::round(static_cast<double>(repetition_rate_hz) * time_buffer_s));
  const double msgs_per_sec   = 1.0 / time_buffer_s;
  const double output_rate_mbs =
      static_cast<double>(rows_init) * cols_per_msg * 4.0 * msgs_per_sec / 1.0e6;
  const auto msg_interval_ns =
      static_cast<int64_t>(std::round(time_buffer_s * 1.0e9));

  if (rows_init < 1) {
    std::cerr << "Computed spatial rows=" << rows_init
              << " is too small. Check fiber length and sampling rate.\n";
    return 1;
  }
  if (cols_per_msg < 1) {
    std::cerr << "Computed cols_per_msg=" << cols_per_msg
              << " is too small. Check repetition rate and time buffer.\n";
    return 1;
  }

  // Mutable row count — updated by resize control messages at runtime
  int rows = rows_init;

  std::cout << "DAS Waterfall Generator\n"
            << "  Fiber:          " << fiber_start_m << " – " << fiber_end_m << " m\n"
            << "  Fiber length:   " << fiber_length_m << " m\n"
            << "  v (in fiber):   " << (V_FIBER / 1.0e6) << " × 10⁶ m/s\n"
            << "  Round-trip:     " << (round_trip_time_s * 1.0e6) << " µs\n"
            << "  Sampling rate:  " << sampling_rate_mhz << " MHz\n"
            << "  Points/segment: " << points_per_segment << "\n"
            << "  Spatial DS:     " << spatial_downsample << "×\n"
            << "  Spatial rows:   " << rows << "\n"
            << "  Repetition rt:  " << repetition_rate_hz << " Hz\n"
            << "  Time buffer:    " << (time_buffer_s * 1000.0) << " ms\n"
            << "  Cols/msg:       " << cols_per_msg << "\n"
            << "  Msgs/sec:       " << msgs_per_sec << "\n"
            << "  Msg interval:   " << (msg_interval_ns / 1.0e6) << " ms\n"
            << "  Output rate:    " << output_rate_mbs << " MB/s\n"
            << std::flush;

  signal(SIGINT, handle_sigint);

  // ---- Random number generator -----------------------------------------

  std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
  std::uniform_int_distribution<int> noise_dist(-127, 127);   // ADC int8 range
  std::uniform_real_distribution<float> amp_dist(0.5f, 1.0f);
  std::uniform_int_distribution<int> row_dist(0, rows - 1);
  std::uniform_int_distribution<int> span_dist(5, std::max(5, rows / 10));
  std::uniform_real_distribution<float> trigger_dist(0.0f, 1.0f);

  // ---- Initialize persistent fiber events ------------------------------
  //
  // Events are anchored to absolute fiber distances (meters) so they stay
  // physically consistent when the spatial resolution changes via resize.

  const int n_fiber_events = 4;
  // Four events at 20/40/60/80% of the fiber segment length
  const std::array<double, 4> EVENT_DIST_M = {
      fiber_start_m + fiber_length_m * 0.2,
      fiber_start_m + fiber_length_m * 0.4,
      fiber_start_m + fiber_length_m * 0.6,
      fiber_start_m + fiber_length_m * 0.8,
  };

  // Project absolute fiber distance to row index for current `rows`
  auto event_row = [&](double dist_m) -> int {
    return static_cast<int>(std::round(
        (dist_m - fiber_start_m) / fiber_length_m * rows));
  };

  std::vector<FiberEvent> fiber_events(n_fiber_events);
  for (int i = 0; i < n_fiber_events; i++) {
    fiber_events[i].row_center     = event_row(EVENT_DIST_M[i]);
    fiber_events[i].row_span       = std::max(3, rows / 20);
    fiber_events[i].base_amplitude = 0.3f + 0.2f * (i % 2);
    fiber_events[i].freq           = 0.02f + 0.01f * i;
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
      if (new_rows >= 4 && new_rows <= 65536 &&
          static_cast<int>(new_rows) != rows) {
        rows = static_cast<int>(new_rows);
        std::cout << "Resize → rows=" << rows << "\n" << std::flush;

        // Reproject fiber events to new row resolution
        for (int i = 0; i < n_fiber_events; i++) {
          fiber_events[i].row_center = event_row(EVENT_DIST_M[i]);
          fiber_events[i].row_span   = std::max(3, rows / 20);
        }
        row_dist  = std::uniform_int_distribution<int>(0, rows - 1);
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

      // Base: int8 ADC noise floor, normalized to float32
      for (int r = 0; r < rows; r++) {
        col[r] = static_cast<float>(noise_dist(rng)) * ADC_NORM;
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
            imp.row_span  = span_dist(rng);
            imp.amplitude = amp_dist(rng);
            imp.age       = 0;
            imp.active    = true;
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

      // Clamp to [-1, 1] (mirrors physical ADC saturation)
      for (int r = 0; r < rows; r++) {
        col[r] = std::max(-1.0f, std::min(1.0f, col[r]));
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
      std::cout << "[perf] msg_id=" << (msg_id - 1)
                << " rows=" << rows
                << " cols=" << cols_per_msg
                << " bytes=" << total_bytes
                << " interval=" << (msg_interval_ns / 1.0e6) << "ms\n"
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
