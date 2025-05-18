from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import webbrowser
import threading
import time
PORT = 8000
HOST = "localhost"
URL = f"http://{HOST}:{PORT}/visualize.html"
def open_browser():
    time.sleep(1)
    webbrowser.open(URL)
threading.Thread(target=open_browser).start()
with TCPServer(("", PORT), SimpleHTTPRequestHandler) as httpd:
    print(f"服务器已启动，自动打开: {URL}")
    print("按 Ctrl+C 停止服务器")
    httpd.serve_forever()