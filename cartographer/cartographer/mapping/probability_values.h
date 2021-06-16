/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CARTOGRAPHER_MAPPING_PROBABILITY_VALUES_H_
#define CARTOGRAPHER_MAPPING_PROBABILITY_VALUES_H_

#include <cmath>
#include <vector>

#include "cartographer/common/math.h"
#include "cartographer/common/port.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

namespace {


/// Float[0.1,0.9] ---> Value[1,32767] 避免浮点数运算
inline uint16 BoundedFloatToValue(const float float_value,
                                  const float lower_bound,
                                  const float upper_bound) {
  const int value =
      common::RoundToInt(
          (common::Clamp(float_value, lower_bound, upper_bound) - lower_bound) *
          (32766.f / (upper_bound - lower_bound))) +
      1;
  // DCHECK for performance.
  DCHECK_GE(value, 1);
  DCHECK_LE(value, 32767);
  return value;
}

}  // namespace

///----Probability p(x=1); CorrespondenceCost p(x=0); Odds p(x=1)/p(x=0)----

inline float Odds(float probability) {
  return probability / (1.f - probability);
}

inline float ProbabilityFromOdds(const float odds) {
  return odds / (odds + 1.f);
}

inline float ProbabilityToCorrespondenceCost(const float probability) {
  return 1.f - probability;
}

inline float CorrespondenceCostToProbability(const float correspondence_cost) {
  return 1.f - correspondence_cost;
}

/// Probability 0.1~0.9; CorrespondenceCost 0.1~0.9
constexpr float kMinProbability = 0.1f;
constexpr float kMaxProbability = 1.f - kMinProbability;
constexpr float kMinCorrespondenceCost = 1.f - kMaxProbability;
constexpr float kMaxCorrespondenceCost = 1.f - kMinProbability;

// Clamps probability to be in the range [kMinProbability, kMaxProbability].
inline float ClampProbability(const float probability) {
  return common::Clamp(probability, kMinProbability, kMaxProbability);
}
// Clamps correspondece cost to be in the range [kMinCorrespondenceCost,
// kMaxCorrespondenceCost].
inline float ClampCorrespondenceCost(const float correspondence_cost) {
  return common::Clamp(correspondence_cost, kMinCorrespondenceCost,
                       kMaxCorrespondenceCost);
}

///-----------------------------FloatToValue-----------------------------------

constexpr uint16 kUnknownProbabilityValue = 0; // 未知的格子 Value设为0
constexpr uint16 kUnknownCorrespondenceValue = kUnknownProbabilityValue;
constexpr uint16 kUpdateMarker = 1u << 15; // 是否进行测量更新标志位(进行hit更新之后, 防止再进行miss更新)
                                           // 一个Cell可能 即是hit又是miss

// Converts a correspondence_cost to a uint16 in the [1, 32767] range.
inline uint16 CorrespondenceCostToValue(const float correspondence_cost) {
  return BoundedFloatToValue(correspondence_cost, kMinCorrespondenceCost,
                             kMaxCorrespondenceCost);
}

// Converts a probability to a uint16 in the [1, 32767] range.
inline uint16 ProbabilityToValue(const float probability) {
  return BoundedFloatToValue(probability, kMinProbability, kMaxProbability);
}

///-----------------------------ValueToFloat(查表)---------------------------------

// 用查表代替计算
extern const std::vector<float>* const kValueToProbability;
extern const std::vector<float>* const kValueToCorrespondenceCost;

// Converts a uint16 (which may or may not have the update marker set) to a
// probability in the range [kMinProbability, kMaxProbability].
inline float ValueToProbability(const uint16 value) {
  return (*kValueToProbability)[value];
}

// Converts a uint16 (which may or may not have the update marker set) to a
// correspondence cost in the range [kMinCorrespondenceCost,
// kMaxCorrespondenceCost].
inline float ValueToCorrespondenceCost(const uint16 value) {
  return (*kValueToCorrespondenceCost)[value];
}

///-----------------------------ValueToValue----------------------------------

inline uint16 ProbabilityValueToCorrespondenceCostValue(
    uint16 probability_value) {

  // 未知的
  if (probability_value == kUnknownProbabilityValue) {
    return kUnknownCorrespondenceValue;
  }

  // 是否超出范围
  bool update_carry = false;
  if (probability_value > kUpdateMarker) {
    probability_value -= kUpdateMarker;
    update_carry = true;
  }

  // Value to Value
  uint16 result = CorrespondenceCostToValue(
      ProbabilityToCorrespondenceCost(ValueToProbability(probability_value)));
  if (update_carry) result += kUpdateMarker; // 超出范围的话还得加回去

  return result;
}

inline uint16 CorrespondenceCostValueToProbabilityValue(
    uint16 correspondence_cost_value) {
  if (correspondence_cost_value == kUnknownCorrespondenceValue)
    return kUnknownProbabilityValue;
  bool update_carry = false;
  if (correspondence_cost_value > kUpdateMarker) {
    correspondence_cost_value -= kUpdateMarker;
    update_carry = true;
  }
  uint16 result = ProbabilityToValue(CorrespondenceCostToProbability(
      ValueToCorrespondenceCost(correspondence_cost_value)));
  if (update_carry) result += kUpdateMarker;
  return result;
}

///---------------------------计算hit_table 和 miss_table----------------------------

std::vector<uint16> ComputeLookupTableToApplyOdds(float odds); // hit_table
std::vector<uint16> ComputeLookupTableToApplyCorrespondenceCostOdds(float odds); // miss_table

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_PROBABILITY_VALUES_H_
