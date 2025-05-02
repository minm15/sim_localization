// src/localizationMain.cpp

#include "sim_local/nuscenes_node.hpp" // your existing class
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    // Allow dataset to be passed in via remap/override
    rclcpp::NodeOptions opts;
    opts.allow_undeclared_parameters(true);
    opts.automatically_declare_parameters_from_overrides(true);
    auto main_node = std::make_shared<rclcpp::Node>("localization_main", opts);

    // Declare & fetch
    // main_node->declare_parameter<std::string>("dataset", "nuscenes");
    std::string dataset = main_node->get_parameter("dataset").as_string();

    if (dataset == "nuscenes") {
        // pass along the same NodeOptions so that all parameter overrides propagate
        auto loc_node = std::make_shared<sim_local::NuscenesNode>(opts);
        rclcpp::spin(loc_node);

    } else {
        RCLCPP_ERROR(main_node->get_logger(),
                     "Unsupported dataset '%s', only 'nuscenes' is implemented", dataset.c_str());
    }

    rclcpp::shutdown();
    return 0;
}
