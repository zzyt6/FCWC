#pragma once
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <utility>

// A measurement and its timestamp are one item; dropping a scan can never
// shift a parallel timestamp queue onto the wrong point cloud.
template<class T> class MeasurementQueue
{
public:
  struct Sample {double stamp; T value;};
  struct Result {bool accepted; size_t dropped;};
  MeasurementQueue(size_t capacity, double horizon) : capacity_(capacity), horizon_(horizon) {}
  Result push(double stamp, T value)
  {
    if (!std::isfinite(stamp) || stamp <= last_stamp_) return {false, 0};
    last_stamp_ = stamp;
    samples_.push_back({stamp, std::move(value)});
    size_t dropped = 0;
    while (samples_.size() > 1 && (samples_.size() > capacity_ ||
      stamp-samples_.front().stamp > horizon_)) {
      samples_.pop_front();
      ++dropped;
    }
    return {true, dropped};
  }
  bool empty() const {return samples_.empty();}
  size_t size() const {return samples_.size();}
  const Sample & front() const {return samples_.front();}
  void pop_front() {samples_.pop_front();}
private:
  size_t capacity_;
  double horizon_;
  double last_stamp_ = -std::numeric_limits<double>::infinity();
  std::deque<Sample> samples_;
};
