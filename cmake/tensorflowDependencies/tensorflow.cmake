add_definitions(-w)

set(TENSORFLOW_INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/deepLearning/tensorflow/bazel-out/host/genfiles/
        ${PROJECT_SOURCE_DIR}/deepLearning/tensorflow/)

set(TENSORFLOW_SHARED_LIBRARY ${PROJECT_SOURCE_DIR}/deepLearning/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so ${PROJECT_SOURCE_DIR}/deepLearning/tensorflow/bazel-bin/tensorflow/libtensorflow.so)

function(use_tensorflow TARGET)
    target_include_directories(${TARGET} PUBLIC ${TENSORFLOW_INCLUDE_DIRS})
    target_link_libraries(${TARGET} ${TENSORFLOW_LIBRARIES})
endfunction()
