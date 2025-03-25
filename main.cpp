#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <openvino/op/ops.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/auto/properties.hpp>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif
using namespace std;
using Clock = std::chrono::high_resolution_clock;

void print_input_and_outputs_info(const ov::Model &network);

int main(int argc, char *argv[])
{
    std::string device_name = argc > 1 ? argv[1] : "";
    std::string model_path1 = argc > 2 ? argv[2] : "";
    std::string model_path2 = argc > 3 ? argv[3] : "";
    std::string model_path3 = argc > 4 ? argv[4] : "";
    int niters = 1000;
    std::atomic<bool> stop_memory_logging(false);
    if (device_name.empty() || (model_path1.empty() && model_path2.empty() && model_path3.empty()))
    {
        std::cout << "usage: " << argv[0]
                  << " <device_name> <model_path1> <model_path2> <model_path3> <number of inferences>" << std::endl;
        std::cerr << "Error: Device name or model paths cannot be empty. Please provide valid inputs." << std::endl;
        return -1;
    }
    std::ostringstream csv_filename;
    csv_filename << device_name << "_";
    if (!model_path1.empty())
    {
        csv_filename << model_path1.substr(model_path1.find_last_of("/\\") + 1) << "_";
    }
    if (!model_path2.empty())
    {
        csv_filename << model_path2.substr(model_path2.find_last_of("/\\") + 1) << "_";
    }
    if (!model_path3.empty())
    {
        csv_filename << model_path3.substr(model_path3.find_last_of("/\\") + 1) << "_";
    }
    csv_filename << "memory_footprint";
    std::string csv_file_path = csv_filename.str();
    std::replace(csv_file_path.begin(), csv_file_path.end(), '/', '_');
    std::replace(csv_file_path.begin(), csv_file_path.end(), ',', '_');
    std::replace(csv_file_path.begin(), csv_file_path.end(), '.', '_');
    std::replace(csv_file_path.begin(), csv_file_path.end(), ':', '_');
    csv_file_path = csv_file_path + ".csv";
    std::cout << "Will write memory usage data to: " << csv_file_path << std::endl;
    std::ofstream csv_file(csv_file_path);
    if (!csv_file.is_open())
    {
        std::cerr << "Error: Unable to open file for writing memory usage data." << std::endl;
        return -1;
    }
    // Write CSV header
    csv_file << "Time (s),Commit Size (MB),Heap Size (MB)" << std::endl;
    std::thread memory_logger([&]()
                              {
        auto start_time = Clock::now();
        int count = 0;
        while (!stop_memory_logging) {
#ifdef _WIN32
            HANDLE process = GetCurrentProcess();
            PROCESS_MEMORY_COUNTERS_EX pmc;
            if (GetProcessMemoryInfo(process, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
                SIZE_T commit_size = pmc.PrivateUsage; // Commit size
                SIZE_T heap_size = pmc.WorkingSetSize; // Heap size
                auto current_time = Clock::now();
                std::chrono::duration<double> elapsed_time = current_time - start_time;
                csv_file << count << "," << commit_size / (1024 * 1024) << "," << heap_size / (1024 * 1024) << std::endl;
                std::cout << "Index: " << count++ << "\tCommit size: " << commit_size / (1024 * 1024) << "\tHeap size: " << heap_size / (1024 * 1024) << std::endl;
            }
#endif
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } });

    ov::Core core;
    std::cout << "OpenVINO version: " << ov::get_openvino_version() << std::endl;
    std::cout << "Device Name: " << device_name << std::endl;
    ov::CompiledModel compiled_model_1, compiled_model_2, compiled_model_3;
    std::shared_ptr<ov::Model> model_1, model_2, model_3;
    ov::AnyMap properties = device_name.find("AUTO") != std::string::npos ? ov::AnyMap{ov::cache_dir("my_caches"), ov::intel_auto::enable_runtime_fallback(false)}
                                                                          : ov::AnyMap{ov::cache_dir("my_caches")};
    // Example: Compile the first model
    if (!model_path1.empty())
    {
        std::cout << "Model Path 1: " << model_path1 << std::endl;
        model_1 = core.read_model(model_path1);
        compiled_model_1 = core.compile_model(model_1, device_name, properties);
        std::cout << "Model 1 compiled successfully." << std::endl;
    }
    if (!model_path2.empty())
    {
        std::cout << "Model Path 2: " << model_path2 << std::endl;
        model_2 = core.read_model(model_path2);
        compiled_model_2 = core.compile_model(model_2, device_name, properties);
        std::cout << "Model 2 compiled successfully." << std::endl;
    }
    if (!model_path3.empty())
    {
        std::cout << "Model Path 3: " << model_path3 << std::endl;
        model_3 = core.read_model(model_path3);
        compiled_model_3 = core.compile_model(model_3, device_name, properties);
        std::cout << "Model 3 compiled successfully." << std::endl;
    }
    std::cout << "Will start inference on model 1 with number of iterations: " << niters << std::endl;

    try
    {
        // will implement inference on model_1 only
        if (!model_1)
        {
            OPENVINO_ASSERT(model_1->inputs().size() == 1, "Sample supports models with 1 input only");
        }
        print_input_and_outputs_info(*model_1);

        // -------- Step 3. Set up input --------
        auto input = model_1->get_parameters().at(0);
        if (input->get_partial_shape().is_dynamic())
        {
            throw std::logic_error("Dynamic models are not supported for this APP.");
        }

        ov::element::Type input_type = input->get_element_type();
        ov::Shape input_shape = input->get_shape();
        ov::Tensor input_tensor = ov::Tensor{input_type, input_shape};

        // -------- Step 5. Create an infer request --------
        ov::InferRequest infer_request = compiled_model_1.create_infer_request();
        infer_request.set_tensor(input, input_tensor);

        // -------- Step 6. Perform inference and calculate FPS --------
        auto start_time = Clock::now();
        for (int i = 0; i < niters; i++)
        {
            auto iter_start_time = Clock::now();
            // Perform inference
            infer_request.infer();
            infer_request.wait();
            if (i % 50 == 0)
            {
                auto current_time = Clock::now();
                std::chrono::duration<double> elapsed_time = current_time - iter_start_time;
                std::cout << "Iteration: " << i << " iteration time: " << elapsed_time.count() << " seconds"
                          << std::endl;
            }
        }
        auto end_time = Clock::now();
        std::chrono::duration<double> total_duration = end_time - start_time;
        double fps = niters / total_duration.count();
        std::cout << "Inference completed. " << "FPS: " << fps << " Total Time: " << total_duration.count()
                  << " seconds" << std::endl;
        csv_file.close();
    }
    catch (const std::exception &ex)
    {
        csv_file.close();
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    stop_memory_logging = true;
    if (memory_logger.joinable())
    {
        memory_logger.join();
    }
    return 0;
}

void print_input_and_outputs_info(const ov::Model &network)
{
    std::cout << "Model Name: " << network.get_name() << std::endl;
    std::cout << "\tInputs:" << std::endl;
    for (auto &&input : network.inputs())
    {
        std::string in_name;
        std::string node_name;

        // Workaround for "tensor has no name" issue
        try
        {
            for (const auto &name : input.get_names())
            {
                in_name += name + " , ";
            }
            in_name = in_name.substr(0, in_name.size() - 3);
        }
        catch (const ov::Exception &)
        {
        }

        try
        {
            node_name = input.get_node()->get_friendly_name();
        }
        catch (const ov::Exception &)
        {
        }

        if (in_name == "")
        {
            in_name = "***NO_NAME***";
        }
        if (node_name == "")
        {
            node_name = "***NO_NAME***";
        }

        std::cout << "\t    " << in_name << " (node: " << node_name << ") : " << input.get_element_type() << " / "
                  << ov::layout::get_layout(input).to_string() << " / " << input.get_partial_shape() << std::endl;
    }

    std::cout << "\tOutputs:" << std::endl;
    for (auto &&output : network.outputs())
    {
        std::string out_name;
        std::string node_name;

        // Workaround for "tensor has no name" issue
        try
        {
            for (const auto &name : output.get_names())
            {
                out_name += name + " , ";
            }
            out_name = out_name.substr(0, out_name.size() - 3);
        }
        catch (const ov::Exception &)
        {
        }
        try
        {
            node_name = output.get_node()->get_input_node_ptr(0)->get_friendly_name();
        }
        catch (const ov::Exception &)
        {
        }

        if (out_name == "")
        {
            out_name = "***NO_NAME***";
        }
        if (node_name == "")
        {
            node_name = "***NO_NAME***";
        }

        std::cout << "\t    " << out_name << " (node: " << node_name << ") : " << output.get_element_type() << " / "
                  << ov::layout::get_layout(output).to_string() << " / " << output.get_partial_shape() << std::endl;
    }
}