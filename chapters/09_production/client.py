from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

import cgi
import classification_service_pb2
from grpc.beta import implementations

class ClientApp(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond_form()

    def respond_form(self, response=""):

        form = """
        <html><body>
        <h1>Image classification service</h1>
        <form enctype="multipart/form-data" method="post">
        <div>Image: <input type="file" name="file" accept="image/jpeg"></div>
        <div><input type="submit" value="Upload"></div>
        </form>
        %s
        </body></html>
        """

        response = form % response

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-length", len(response))
        self.end_headers()
        self.wfile.write(response)

    def do_POST(self):

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': self.headers['Content-Type'],
            })

        request = classification_service_pb2.ClassificationRequest()
        request.input = form['file'].file.read()

        channel = implementations.insecure_channel("127.0.0.1", 9999)
        stub = classification_service_pb2.beta_create_ClassificationService_stub(channel)
        response = stub.classify(request, 10) # 10 secs timeout

        self.respond_form("<div>Response: %s</div>" % response)


if __name__ == '__main__':
    host_port = ('0.0.0.0', 8080)
    print "Serving in %s:%s" % host_port
    HTTPServer(host_port, ClientApp).serve_forever()