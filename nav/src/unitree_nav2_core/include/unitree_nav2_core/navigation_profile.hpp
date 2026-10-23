#ifndef UNITREE_NAV2_CORE__NAVIGATION_PROFILE_HPP_
#define UNITREE_NAV2_CORE__NAVIGATION_PROFILE_HPP_

#include <cmath>
#include <string>
#include <vector>

#include "nav2_util/node_utils.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/parameter.hpp"

namespace unitree_nav2_core
{

// The web bridge sets <controller>.navigation_speed{,_mode} in one atomic
// transaction for every controller; each controller validates its own pair.
template<typename NodeT>
void declareNavigationProfile(const NodeT & node, const std::string & ns)
{
  nav2_util::declare_parameter_if_not_declared(
    node, ns + ".navigation_speed_mode", rclcpp::ParameterValue(std::string("auto")));
  nav2_util::declare_parameter_if_not_declared(
    node, ns + ".navigation_speed", rclcpp::ParameterValue(0.35));
}

inline rcl_interfaces::msg::SetParametersResult validateNavigationProfile(
  const std::vector<rclcpp::Parameter> & parameters, const std::string & ns)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto & parameter : parameters) {
    if (parameter.get_name() == ns + ".navigation_speed") {
      if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE ||
        !std::isfinite(parameter.as_double()) || parameter.as_double() < 0.20 ||
        parameter.as_double() > 0.60)
      {
        result.successful = false;
        result.reason = "Navigation speed must be between 0.20 and 0.60 m/s";
      }
    } else if (parameter.get_name() == ns + ".navigation_speed_mode") {
      if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING ||
        (parameter.as_string() != "auto" && parameter.as_string() != "fixed"))
      {
        result.successful = false;
        result.reason = "Navigation speed mode must be auto or fixed";
      }
    }
  }
  return result;
}

}  // namespace unitree_nav2_core
#endif
