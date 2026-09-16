#include <bit>
#include <cctype>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <vkml/vkml.hpp>

using Catch::Matchers::ContainsSubstring;

namespace {

// CI sets this so a missing validation layer fails loudly instead of the
// suite silently running without it.
bool validation_required() {
    const char* v = std::getenv("VKML_REQUIRE_VALIDATION");
    return v != nullptr && std::string_view(v) == "1";
}

}  // namespace

TEST_CASE("Context selects a device that meets the Vulkan minimums", "[context]") {
    vkml::Context context;
    const vkml::DeviceInfo& d = context.device_info();

    CHECK_FALSE(d.name.empty());
    CHECK((d.api_version.major > 1 || d.api_version.minor >= 2));

    // Guaranteed minimums from the Vulkan spec's "Required Limits" table.
    CHECK(d.max_workgroup_invocations >= 128);
    CHECK(d.max_workgroup_size[0] >= 128);
    CHECK(d.max_workgroup_count[0] >= 65535);
    CHECK(d.max_shared_memory_bytes >= 16384);
    CHECK(d.max_push_constant_bytes >= 128);
    CHECK(d.max_storage_buffer_range >= (1u << 27));

    CHECK(std::has_single_bit(d.subgroup_size));
    CHECK(std::has_single_bit(d.min_storage_buffer_offset_alignment));
    CHECK(std::has_single_bit(d.non_coherent_atom_size));
    CHECK(d.device_local_bytes > 0);

    if (validation_required()) CHECK(d.validation_enabled);
    CHECK(context.validation_error_count() == 0);
}

TEST_CASE("An unknown device name is rejected and the error lists what exists", "[context]") {
    vkml::ContextOptions options;
    options.device_name = "no-such-gpu-xyz";
    REQUIRE_THROWS_WITH(vkml::Context{options},
                        ContainsSubstring("no-such-gpu-xyz") && ContainsSubstring("available:"));
}

TEST_CASE("Selecting by name is case-insensitive", "[context]") {
    std::string name;
    {
        vkml::Context context;
        name = context.device_info().name;
    }
    for (char& c : name) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

    vkml::ContextOptions options;
    options.device_name = name;
    vkml::Context context{options};
    CHECK(context.device_info().name.size() == name.size());
}

TEST_CASE("Only one Context may be alive at a time", "[context]") {
    {
        vkml::Context first;
        REQUIRE_THROWS_WITH(vkml::Context{}, ContainsSubstring("already alive"));
    }
    // The rejected attempt must not leave the one-per-process guard stuck.
    vkml::Context again;
    CHECK(again.validation_error_count() == 0);
}

TEST_CASE("A Context that fails to construct releases the device guard", "[context]") {
    vkml::ContextOptions bad;
    bad.device_name = "no-such-gpu-xyz";
    REQUIRE_THROWS(vkml::Context{bad});

    vkml::Context good;
    CHECK(good.validation_error_count() == 0);
}

TEST_CASE("Moving a Context transfers ownership of the device", "[context]") {
    vkml::Context a;
    const std::string name = a.device_info().name;
    vkml::Context b = std::move(a);
    CHECK(b.device_info().name == name);
}
