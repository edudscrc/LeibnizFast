/*
 * generator.cpp — Spatial-temporal data generator → ZeroMQ PUSH
 *
 * Simulates a sensor array acquiring spatial-temporal data and streaming
 * float32 columns over ZMQ. A Python bridge (bridge.py) relays messages
 * to WebSocket clients.
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
 *   - Sensor acquires samples at `sampling_rate` MHz over a spatial extent
 *   - Samples normalized to float32 range [-0.992, 0.992]
 *   - Spatial downsampling by `spatial_downsample` (integer step)
 *   - rows = ceil(round(sampling_rate_MHz × 1e6 × 2 × spatial_extent /
 * propagation_velocity) / spatial_downsample)
 *   - cols_per_msg = round(repetition_rate_Hz × time_buffer_s)
 *   - msg_interval_ns = round(time_buffer_s × 1e9)
 *
 * Build:
 *   g++ -std=c++17 -O2 -o generator generator.cpp -lzmq
 *
 * Run:
 *   ./generator                                                    # defaults
 *   ./generator --spatial-start 5000 --spatial-end 15000           # custom
 * spatial range
 *   ./generator --sampling-rate 200 --spatial-downsample 10        # lower
 * resolution
 *   ./generator --repetition-rate 5000 --time-buffer 0.4           # slower
 * rate, larger batches
 *   ./generator --debug                                            #
 * per-message timing
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
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ---- Constants -----------------------------------------------------------

static constexpr uint32_t WATERFALL_MAGIC = 0x4C465A10u;
static constexpr int HEADER_BYTES = 16;  // 4 × uint32

// Propagation velocity for round-trip spatial sampling (m/s)
static constexpr double PROPAGATION_VELOCITY = 2.0432e8;


// ---- Signal handling -----------------------------------------------------

static volatile bool g_running = true;
static void handle_sigint(int) { g_running = false; }


// ---- Main ----------------------------------------------------------------

int main(int argc, char* argv[]) {
  double spatial_start_m = 10000.0;  // meters
  double spatial_end_m = 15000.0;    // meters
  int sampling_rate_mhz = 400;       // MHz
  int repetition_rate_hz = 1000;     // Hz
  int spatial_downsample = 5;        // integer step
  double time_buffer_s = 0.15;       // seconds
  bool debug = false;

  for (int a = 1; a < argc; ++a) {
    const std::string arg(argv[a]);
    if (arg == "--spatial-start" && a + 1 < argc) {
      spatial_start_m = std::stod(argv[++a]);
      if (spatial_start_m < 0.0 || spatial_start_m > 1.0e6) {
        std::cerr << "spatial-start must be 0..1000000 m\n";
        return 1;
      }
    } else if (arg == "--spatial-end" && a + 1 < argc) {
      spatial_end_m = std::stod(argv[++a]);
      if (spatial_end_m < 1.0 || spatial_end_m > 1.0e6) {
        std::cerr << "spatial-end must be 1..1000000 m\n";
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

  if (spatial_end_m <= spatial_start_m) {
    std::cerr << "spatial-end must be greater than spatial-start\n";
    return 1;
  }

  // ---- Spatial sampling computation --------------------------------------

  const double spatial_extent_m = spatial_end_m - spatial_start_m;
  const double round_trip_time_s =
      2.0 * spatial_extent_m / PROPAGATION_VELOCITY;
  const int points_per_segment = static_cast<int>(
      std::round(sampling_rate_mhz * 1.0e6 * round_trip_time_s));
  const int rows_init = static_cast<int>(
      std::ceil(static_cast<double>(points_per_segment) / spatial_downsample));
  const int cols_per_msg = static_cast<int>(
      std::round(static_cast<double>(repetition_rate_hz) * time_buffer_s));
  const double msgs_per_sec = 1.0 / time_buffer_s;
  const double output_rate_mbs = static_cast<double>(rows_init) * cols_per_msg *
                                 4.0 * msgs_per_sec / 1.0e6;
  const auto msg_interval_ns =
      static_cast<int64_t>(std::round(time_buffer_s * 1.0e9));

  if (rows_init < 1) {
    std::cerr << "Computed spatial rows=" << rows_init
              << " is too small. Check spatial extent and sampling rate.\n";
    return 1;
  }
  if (cols_per_msg < 1) {
    std::cerr << "Computed cols_per_msg=" << cols_per_msg
              << " is too small. Check repetition rate and time buffer.\n";
    return 1;
  }

  // Mutable row count — updated by resize control messages at runtime
  int rows = rows_init;

  std::cout << "Waterfall Generator\n"
            << "  Spatial range:  " << spatial_start_m << " – " << spatial_end_m
            << " m\n"
            << "  Spatial extent: " << spatial_extent_m << " m\n"
            << "  Propagation v:  " << (PROPAGATION_VELOCITY / 1.0e6)
            << " × 10\u2076 m/s\n"
            << "  Round-trip:     " << (round_trip_time_s * 1.0e6)
            << " \u00b5s\n"
            << "  Sampling rate:  " << sampling_rate_mhz << " MHz\n"
            << "  Points/segment: " << points_per_segment << "\n"
            << "  Spatial DS:     " << spatial_downsample << "\u00d7\n"
            << "  Spatial rows:   " << rows << "\n"
            << "  Repetition rt:  " << repetition_rate_hz << " Hz\n"
            << "  Time buffer:    " << (time_buffer_s * 1000.0) << " ms\n"
            << "  Cols/msg:       " << cols_per_msg << "\n"
            << "  Msgs/sec:       " << msgs_per_sec << "\n"
            << "  Msg interval:   " << (msg_interval_ns / 1.0e6) << " ms\n"
            << "  Output rate:    " << output_rate_mbs << " MB/s\n"
            << std::flush;

  signal(SIGINT, handle_sigint);

  // ---- Random number generator (light noise only) ----------------------

  std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
  std::uniform_real_distribution<float> noise_dist(-0.03f, 0.03f);

  // ---- Sine-wave sweep parameters --------------------------------------
  //
  // Three sine waves sweep a Gaussian peak up and down through the spatial
  // axis at different frequencies, creating a pleasant interference pattern.
  // Angular frequencies are chosen so the pattern never quite repeats.

  // Primary wave: slow, wide sweep across ~70% of the spatial extent
  static constexpr float WAVE1_FREQ = 0.003f;   // radians per column
  static constexpr float WAVE1_AMP  = 0.35f;    // fraction of rows (peak-to-peak/2)
  static constexpr float WAVE1_WIDTH = 0.04f;   // Gaussian width as fraction of rows
  static constexpr float WAVE1_BRIGHTNESS = 0.9f;

  // Secondary wave: faster, narrower, dimmer
  static constexpr float WAVE2_FREQ = 0.0073f;
  static constexpr float WAVE2_AMP  = 0.25f;
  static constexpr float WAVE2_WIDTH = 0.025f;
  static constexpr float WAVE2_BRIGHTNESS = 0.6f;

  // Tertiary wave: very slow drift, wide and faint (background glow)
  static constexpr float WAVE3_FREQ = 0.0011f;
  static constexpr float WAVE3_AMP  = 0.40f;
  static constexpr float WAVE3_WIDTH = 0.08f;
  static constexpr float WAVE3_BRIGHTNESS = 0.3f;

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

  using clk = std::chrono::steady_clock;
  using dur_ms = std::chrono::duration<double, std::milli>;

  // Helper: milliseconds between two time points
  auto ms_between = [](clk::time_point a, clk::time_point b) -> double {
    return std::chrono::duration_cast<dur_ms>(b - a).count();
  };

  auto next_send_time = clk::now();
  auto t_last_iter = clk::now();  // tracks when the previous iteration ended

  while (g_running) {
    auto t_iter_start = clk::now();  // wall time at start of this iteration

    // ---- Check for control messages (non-blocking) -------------------

    uint8_t ctrl_buf[4];
    int ctrl_len =
        zmq_recv(sock_ctrl, ctrl_buf, sizeof(ctrl_buf), ZMQ_DONTWAIT);
    if (ctrl_len == 4) {
      uint32_t new_rows;
      memcpy(&new_rows, ctrl_buf, 4);
      if (new_rows >= 4 && new_rows <= 65536 &&
          static_cast<int>(new_rows) != rows) {
        rows = static_cast<int>(new_rows);
        std::cout << "Resize \u2192 rows=" << rows << "\n" << std::flush;

        // Reallocate buffers for the new row count
        const size_t new_payload =
            static_cast<size_t>(rows) * cols_per_msg * sizeof(float);
        send_buf.resize(HEADER_BYTES + new_payload);
        col_buf.resize(static_cast<size_t>(rows) * cols_per_msg);
      }
    }

    // ---- Generate column data ------------------------------------------

    auto t_gen_start = clk::now();

    for (int c = 0; c < cols_per_msg; c++) {
      float* col = col_buf.data() + static_cast<size_t>(c) * rows;
      const float t = static_cast<float>(col_index + c);
      const float half = static_cast<float>(rows) * 0.5f;

      // Sine-wave centers (oscillate around the midpoint of the spatial axis)
      const float center1 = half + half * WAVE1_AMP * std::sin(t * WAVE1_FREQ);
      const float center2 = half + half * WAVE2_AMP * std::sin(t * WAVE2_FREQ + 1.0f);
      const float center3 = half + half * WAVE3_AMP * std::sin(t * WAVE3_FREQ + 2.5f);

      const float inv_w1 = 1.0f / std::max(1.0f, static_cast<float>(rows) * WAVE1_WIDTH);
      const float inv_w2 = 1.0f / std::max(1.0f, static_cast<float>(rows) * WAVE2_WIDTH);
      const float inv_w3 = 1.0f / std::max(1.0f, static_cast<float>(rows) * WAVE3_WIDTH);

      for (int r = 0; r < rows; r++) {
        const float fr = static_cast<float>(r);

        // Three Gaussian peaks sweeping at different speeds
        const float d1 = (fr - center1) * inv_w1;
        const float d2 = (fr - center2) * inv_w2;
        const float d3 = (fr - center3) * inv_w3;

        float val = WAVE1_BRIGHTNESS * std::exp(-d1 * d1) +
                    WAVE2_BRIGHTNESS * std::exp(-d2 * d2) +
                    WAVE3_BRIGHTNESS * std::exp(-d3 * d3) +
                    noise_dist(rng);

        col[r] = std::max(-1.0f, std::min(1.0f, val));
      }
    }

    col_index += cols_per_msg;

    auto t_gen_end = clk::now();

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

    auto t_build_end = clk::now();

    // ---- Send ----------------------------------------------------------

    const size_t total_bytes = HEADER_BYTES + payload_bytes;
    int rc = zmq_send(sock_data, send_buf.data(), total_bytes, ZMQ_DONTWAIT);

    auto t_send_end = clk::now();

    // ---- Throttle to target rate ---------------------------------------

    next_send_time += std::chrono::nanoseconds(msg_interval_ns);
    if (next_send_time > t_send_end) {
      std::this_thread::sleep_until(next_send_time);
    } else {
      // Falling behind — reset to avoid burst catchup
      next_send_time = t_send_end;
    }

    auto t_sleep_end = clk::now();

    if (debug) {
      const double gap_ms = ms_between(t_last_iter, t_iter_start);
      const double gen_ms = ms_between(t_gen_start, t_gen_end);
      const double build_ms = ms_between(t_gen_end, t_build_end);
      const double zmq_send_ms = ms_between(t_build_end, t_send_end);
      const double sleep_ms = ms_between(t_send_end, t_sleep_end);
      const double total_ms = ms_between(t_iter_start, t_sleep_end);
      std::cout << "[perf] msg_id=" << (msg_id - 1) << std::fixed
                << std::setprecision(2) << " gap=" << gap_ms << "ms"
                << " gen=" << gen_ms << "ms"
                << " build=" << build_ms << "ms"
                << " zmq_send=" << zmq_send_ms << "ms"
                << " sleep=" << sleep_ms << "ms"
                << " total=" << total_ms << "ms"
                << " bytes=" << total_bytes << (rc < 0 ? " [DROPPED]" : "")
                << "\n"
                << std::flush;
    }

    t_last_iter = t_sleep_end;
  }

  // ---- Cleanup ---------------------------------------------------------

  std::cout << "\nShutting down...\n";
  zmq_close(sock_data);
  zmq_close(sock_ctrl);
  zmq_ctx_destroy(ctx);
  return 0;
}
