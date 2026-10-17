import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.Socket;

// 实验5.3：TCP文件接收客户端
public class FileTransferClient {
    // 默认服务端地址与端口
    private static final String DEFAULT_HOST = "127.0.0.1";
    private static final int DEFAULT_PORT = 7000;

    public static void main(String[] args) {
        String host = DEFAULT_HOST;
        int port = DEFAULT_PORT;
        String saveName = null;

        // 参数格式: [host] [port] [saveName]
        if (args.length >= 1) {
            host = args[0];
        }
        if (args.length >= 2) {
            port = Integer.parseInt(args[1]);
        }
        if (args.length >= 3) {
            saveName = args[2];
        }

        System.out.println("客户端连接服务器: " + host + ":" + port);

        try (Socket socket = new Socket(host, port);
             DataOutputStream out = new DataOutputStream(socket.getOutputStream());
             DataInputStream in = new DataInputStream(socket.getInputStream())) {

            // 简单应用层协议：先发送请求口令
            out.writeUTF("REQUEST_FILE");
            out.flush();
            System.out.println("发送请求: REQUEST_FILE");

            String status = in.readUTF();
            System.out.println("接收服务端状态: " + status);
            // 非OK状态直接终止，不再尝试读文件体
            if (!"OK".equalsIgnoreCase(status)) {
                System.out.println("服务端拒绝传输，客户端结束");
                return;
            }

            String remoteFileName = in.readUTF();
            long fileSize = in.readLong();
            System.out.println("接收文件名: " + remoteFileName);
            System.out.println("接收文件大小: " + fileSize + " 字节");

            String localName;
            if (saveName != null && !saveName.isEmpty()) {
                localName = saveName;
            } else {
                // 未指定时使用默认前缀，避免覆盖同名文件
                localName = "received_" + remoteFileName;
            }

            File saveDir = new File("downloaded");
            if (!saveDir.exists()) {
                saveDir.mkdirs();
            }
            File outFile = new File(saveDir, localName);

            try (BufferedOutputStream fileOut = new BufferedOutputStream(new FileOutputStream(outFile))) {
                byte[] buffer = new byte[4096];
                long remaining = fileSize;
                long received = 0;

                // 按服务端声明的长度接收，防止多读
                while (remaining > 0) {
                    int readLen = in.read(buffer, 0, (int) Math.min(buffer.length, remaining));
                    if (readLen == -1) {
                        break;
                    }
                    fileOut.write(buffer, 0, readLen);
                    remaining -= readLen;
                    received += readLen;
                }
                fileOut.flush();
                System.out.println("文件接收完成，已保存为: " + outFile.getAbsolutePath());
                System.out.println("实际接收大小: " + received + " 字节");
            }
        } catch (IOException e) {
            System.err.println("客户端异常: " + e.getMessage());
            e.printStackTrace();
            return;
        }

        System.out.println("客户端退出");
    }
}
