

add_custom_target(active-offload-target DEPENDS active-offload-target)
add_dependencies(clang active-offload-target)

install(PROGRAMS active-offload-target
	DESTINATION bin
	COMPONENT active-offload-target)

# Install symbolic links to active-offload-target
install(PROGRAMS
    ${CMAKE_CURRENT_SOURCE_DIR}/amdgpu-arch
    ${CMAKE_CURRENT_SOURCE_DIR}/nvptx-arch
    DESTINATION bin)
