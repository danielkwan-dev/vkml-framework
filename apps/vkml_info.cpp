// Prints the device vkml selects and the limits its kernels are tuned against.
//
//   vkml-info                 auto-select (or honour VKML_DEVICE)
//   vkml-info --device intel  pick the first device whose name contains "intel"

#include <cstdio>
#include <exception>
#include <string_view>

#include <vkml/vkml.hpp>

namespace {

double gib(std::uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

}  // namespace

int main(int argc, char** argv) {
    vkml::ContextOptions options;
    if (argc == 3 && std::string_view(argv[1]) == "--device") {
        options.device_name = argv[2];
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--device <name substring>]\n", argv[0]);
        return 2;
    }

    try {
        vkml::Context context{options};
        const vkml::DeviceInfo& d = context.device_info();
        const auto type = vkml::to_string(d.type);

        std::printf("device           %s\n", d.name.c_str());
        std::printf("type             %.*s (%s)\n", static_cast<int>(type.size()), type.data(),
                    d.vendor.c_str());
        std::printf("driver           %s\n", d.driver.c_str());
        std::printf("vulkan           %u.%u.%u\n", d.api_version.major, d.api_version.minor,
                    d.api_version.patch);
        std::printf("unified memory   %s\n", d.unified_memory ? "yes" : "no");
        std::printf("device-local     %.2f GiB\n", gib(d.device_local_bytes));
        std::printf("subgroup size    %u\n", d.subgroup_size);
        std::printf("int8 dot product %s\n", d.integer_dot_product ? "accelerated" : "no");
        std::printf("workgroup        %u invocations max, size %u x %u x %u\n",
                    d.max_workgroup_invocations, d.max_workgroup_size[0], d.max_workgroup_size[1],
                    d.max_workgroup_size[2]);
        std::printf("dispatch         %u x %u x %u workgroups max\n", d.max_workgroup_count[0],
                    d.max_workgroup_count[1], d.max_workgroup_count[2]);
        std::printf("shared memory    %u KiB\n", d.max_shared_memory_bytes / 1024);
        std::printf("push constants   %u bytes\n", d.max_push_constant_bytes);
        std::printf("storage buffer   %.2f GiB max range, offsets aligned to %llu bytes\n",
                    gib(d.max_storage_buffer_range),
                    static_cast<unsigned long long>(d.min_storage_buffer_offset_alignment));
        std::printf("non-coherent     %llu-byte flush granularity\n",
                    static_cast<unsigned long long>(d.non_coherent_atom_size));
        std::printf("validation       %s\n", d.validation_enabled ? "on" : "off");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "vkml-info: %s\n", e.what());
        return 1;
    }
    return 0;
}
