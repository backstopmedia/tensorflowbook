#include <iostream>
#include <memory>
#include <string>

#include <grpc++/grpc++.h>

#include "classification_service.grpc.pb.h"

#include "tensorflow_serving/servables/tensorflow/session_bundle_factory.h"

using namespace std;
using namespace tensorflow::serving;
using namespace grpc;

unique_ptr<SessionBundle> createSessionBundle(const string& pathToExportFiles) {
	SessionBundleConfig session_bundle_config = SessionBundleConfig();
	unique_ptr<SessionBundleFactory> bundle_factory;
    SessionBundleFactory::Create(session_bundle_config, &bundle_factory);

	unique_ptr<SessionBundle> sessionBundle;
	bundle_factory->CreateSessionBundle(pathToExportFiles, &sessionBundle);

	return sessionBundle;
}


class ClassificationServiceImpl final : public ClassificationService::Service {

  private:
	unique_ptr<SessionBundle> sessionBundle;

  public:
    ClassificationServiceImpl(unique_ptr<SessionBundle> sessionBundle) :
        sessionBundle(move(sessionBundle)) {};

    Status classify(ServerContext* context, const ClassificationRequest* request,
                    ClassificationResponse* response) override {

		// Load classification signature
		ClassificationSignature signature;
		const tensorflow::Status signatureStatus =
		  GetClassificationSignature(sessionBundle->meta_graph_def, &signature);

		if (!signatureStatus.ok()) {
			return Status(StatusCode::INTERNAL, signatureStatus.error_message());
		}

		// Transform protobuf input to inference input tensor.
		tensorflow::Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
		input.scalar<string>()() = request->input();

		vector<tensorflow::Tensor> outputs;

		// Run inference.
		const tensorflow::Status inferenceStatus = sessionBundle->session->Run(
			{{signature.input().tensor_name(), input}},
			{signature.classes().tensor_name(), signature.scores().tensor_name()},
			{},
			&outputs);

		if (!inferenceStatus.ok()) {
			return Status(StatusCode::INTERNAL, inferenceStatus.error_message());
		}

		// Transform inference output tensor to protobuf output.
		for (int i = 0; i < outputs[0].NumElements(); ++i) {
			ClassificationClass *classificationClass = response->add_classes();
			classificationClass->set_name(outputs[0].flat<string>()(i));
			classificationClass->set_score(outputs[1].flat<float>()(i));
		}

        return Status::OK;

    }
};


int main(int argc, char** argv) {

    if (argc < 3) {
    	cerr << "Usage: server <port> /path/to/export/files" << endl;
		return 1;
    }

	const string serverAddress(string("0.0.0.0:") + argv[1]);
	const string pathToExportFiles(argv[2]);

	unique_ptr<SessionBundle> sessionBundle = createSessionBundle(pathToExportFiles);

	ClassificationServiceImpl classificationServiceImpl(move(sessionBundle));

    ServerBuilder builder;
    builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials());
    builder.RegisterService(&classificationServiceImpl);

    unique_ptr<Server> server = builder.BuildAndStart();
    cout << "Server listening on " << serverAddress << endl;

    server->Wait();

    return 0;
}
