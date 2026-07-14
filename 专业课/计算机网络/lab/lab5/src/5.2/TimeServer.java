import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

// 实验5.2：TCP时间服务端（单客户端版本）
public class TimeServer {
    // 默认监听端口与时间格式
    private static final int DEFAULT_PORT = 5000;
    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public static void main(String[] args) {
        int port = DEFAULT_PORT;
        if (args.length > 0) {
            port = Integer.parseInt(args[0]);
        }

        System.out.println("服务器启动，监听端口: " + port);

        try (ServerSocket serverSocket = new ServerSocket(port)) {
            // 阻塞等待客户端连接
            Socket clientSocket = serverSocket.accept();
            System.out.println("创建客户连接: " + clientSocket.getInetAddress().getHostAddress() + ":" + clientSocket.getPort());

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                 PrintWriter writer = new PrintWriter(clientSocket.getOutputStream(), true)) {

                String command;
                // 循环处理客户端命令，直到对端断开或收到Exit
                while ((command = reader.readLine()) != null) {
                    System.out.println("接收到客户端命令: " + command);

                    String response;
                    if ("Time".equalsIgnoreCase(command)) {
                        response = "服务器当前时间为: " + LocalDateTime.now().format(FORMATTER);
                    } else if ("Exit".equalsIgnoreCase(command)) {
                        // 收到退出指令后回复Bye并结束会话
                        response = "Bye";
                        writer.println(response);
                        System.out.println("发送给客户端: " + response);
                        break;
                    } else {
                        response = "Unknown Command";
                    }

                    writer.println(response);
                    System.out.println("发送给客户端: " + response);
                }
            }

            System.out.println("服务器退出");
        } catch (IOException e) {
            System.err.println("服务器异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
