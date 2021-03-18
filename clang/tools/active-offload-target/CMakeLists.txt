

add_custom_target(active-offload-target DEPENDS active-offload-target)
add_dependencies(clang active-offload-target)

install(PROGRAMS active-offload-target
	DESTINATION bin
	COMPONENT active-offload-target)
