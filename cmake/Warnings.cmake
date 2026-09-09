function(vkml_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4 /permissive-
            $<$<BOOL:${VKML_WERROR}>:/WX>)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
            $<$<BOOL:${VKML_WERROR}>:-Werror>)
    endif()
endfunction()
