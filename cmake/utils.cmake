function(add_target target)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PUBLIC ${DNNL_INCLUDE_DIR} ${SYCL_INCLUDE_DIR} ${ONNX_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIRS} ${GUROBI_INCLUDE_DIR} ${CBC_INCLUDE_DIR} ${CBC_EXAMPLES_DIR})
    target_link_libraries(${target} ${DNNL_LIBRARY} ${SYCL_LIBRARY} ${ONNX_LIBRARY} ${ONNX_PROTO_LIBRARY} ${PROTOBUF_LIBRARY} PNG::PNG ${GUROBI_LIBRARIES} ${CBC_LIBRARIES})
if (${HAS_CBC})
    add_definitions( -DHAS_CBC )
endif()
if (${HAS_GUROBI})
    add_definitions( -DHAS_GUROBI )
endif()
endfunction()
