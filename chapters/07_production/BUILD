py_binary(
    name = "export",
    srcs = [
        "export.py",
    ],
    deps = [
        "@tf_serving//tensorflow_serving/session_bundle:exporter",
        "@org_tensorflow//tensorflow:tensorflow_py",
        # only needed for inception model export
        "@inception_model//inception",
    ],
)

load("@protobuf//:protobuf.bzl", "cc_proto_library")

cc_proto_library(
    name="classification_service_proto",
    srcs=["classification_service.proto"],
    cc_libs = ["@protobuf//:protobuf"],
    protoc="@protobuf//:protoc",
    default_runtime="@protobuf//:protobuf",
    use_grpc_plugin=1
)

cc_binary(
    name = "server",
    srcs = [
        "server.cc",
        ],
    deps = [
        ":classification_service_proto",
        "@tf_serving//tensorflow_serving/servables/tensorflow:session_bundle_factory",
        "@grpc//:grpc++",
        ],
)