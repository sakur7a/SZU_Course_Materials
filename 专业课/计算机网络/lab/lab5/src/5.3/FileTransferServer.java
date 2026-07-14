import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

// 实验5.3：TCP文件发送服务端（单客户端版本）
public class FileTransferServer {
    private static final int DEFAULT_PORT = 7000;

    public static void main(String[] args) {
        int port = DEFAULT_PORT;
        String filePath = "src/5.3/server_data.txt";

        // 参数格式: [port] [filePath]
        if (args.length >= 1) {
            port = Integer.parseInt(args[0]);
        }
        if (args.length >= 2) {
            filePath = args[1];
        }

        File file = new File(filePath);
        // 启动前校验目标文件可用性
        if (!file.exists() || !file.isFile()) {
            System.err.println("服务器错误: 要发送的文件不存在 -> " + file.getAbsolutePath());
            return;
        }

        System.out.println("服务端启动，监听端口: " + port);
        System.out.println("准备发送文件: " + file.getAbsolutePath());

        try (ServerSocket serverSocket = new ServerSocket(port);
               // 阻塞等待一个客户端连接
             Socket clientSocket = serverSocket.accept();
             DataInputStream in = new DataInputStream(clientSocket.getInputStream());
             DataOutputStream out = new DataOutputStream(clientSocket.getOutputStream());
             BufferedInputStream fileIn = new BufferedInputStream(new FileInputStream(file))) {

            System.out.println("客户端已连接: " + clientSocket.getInetAddress().getHostAddress() + ":" + clientSocket.getPort());

            String request = in.readUTF();
            System.out.println("接收客户端请求: " + request);

            // 仅接受约定请求口令
            if (!"REQUEST_FILE".equalsIgnoreCase(request)) {
                out.writeUTF("ERROR");
                out.flush();
                System.out.println("发送响应: ERROR");
                return;
            }

            out.writeUTF("OK");
            out.writeUTF(file.getName());
            out.writeLong(file.length());
            out.flush();
            System.out.println("发送文件名: " + file.getName());
            System.out.println("发送文件大小: " + file.length() + " 字节");

            byte[] buffer = new byte[4096];
            int len;
            long sent = 0;
            // 按块发送文件字节流
            while ((len = fileIn.read(buffer)) != -1) {
                out.write(buffer, 0, len);
                sent += len;
            }
            out.flush();
            System.out.println("文件传输完成，已发送: " + sent + " 字节");
        } catch (IOException e) {
            System.err.println("服务端异常: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("服务端退出");
    }
}
