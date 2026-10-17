import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.Scanner;

// 实验5.2：TCP时间客户端
public class TimeClient {
    // 默认连接地址与端口，可通过命令行参数覆盖
    private static final String DEFAULT_HOST = "127.0.0.1";
    private static final int DEFAULT_PORT = 5000;

    public static void main(String[] args) {
        String host = DEFAULT_HOST;
        int port = DEFAULT_PORT;
        int commandStartIndex = 0;

        // 参数格式: [host] [port] [command1 command2 ...]
        if (args.length >= 1) {
            host = args[0];
            commandStartIndex = 1;
        }
        if (args.length >= 2) {
            port = Integer.parseInt(args[1]);
            commandStartIndex = 2;
        }

        try (Socket socket = new Socket(host, port);
             BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
             PrintWriter writer = new PrintWriter(socket.getOutputStream(), true)) {

            System.out.println("已连接服务器: " + host + ":" + port);

            if (args.length > commandStartIndex) {
                // 批处理模式：直接按参数顺序发送命令
                for (int i = commandStartIndex; i < args.length; i++) {
                    String command = args[i];
                    writer.println(command);
                    System.out.println("发送命令: " + command);

                    String response = reader.readLine();
                    System.out.println("接收消息: " + response);

                    if ("Exit".equalsIgnoreCase(command) || "Bye".equalsIgnoreCase(response)) {
                        break;
                    }
                }
            } else {
                // 交互模式：从控制台循环读取命令
                Scanner scanner = new Scanner(System.in);
                while (true) {
                    System.out.print("请输入命令(Time/Exit): ");
                    String command = scanner.nextLine();
                    writer.println(command);
                    System.out.println("发送命令: " + command);

                    String response = reader.readLine();
                    System.out.println("接收消息: " + response);

                    // 任一方进入结束语义时退出循环
                    if ("Exit".equalsIgnoreCase(command) || "Bye".equalsIgnoreCase(response)) {
                        break;
                    }
                }
            }

            System.out.println("客户端退出");
        } catch (IOException e) {
            System.err.println("客户端异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
