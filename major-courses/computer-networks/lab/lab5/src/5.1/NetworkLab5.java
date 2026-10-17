import java.io.BufferedInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.URL;

// 实验5.1：基础网络编程示例（主机信息、DNS解析、网页下载）
public class NetworkLab5 {
    public static void main(String[] args) {
        try {
            // 任务1：获取本机主机名与IP
            printLocalHostInfo();
            // 任务2：查询目标域名对应的全部IP（可能包含多个A记录）
            printAllIpsOfCsdn();

            // 任务3：下载网页到本地文件并统计字节数
            String targetUrl = "http://www.szu.edu.cn";
            String outputFile = "szu_homepage.html";
            long size = downloadWebPage(targetUrl, outputFile);

            System.out.println("\n下载网页并统计大小");
            System.out.println("下载地址: " + targetUrl);
            System.out.println("保存文件: " + outputFile);
            System.out.println("文件大小: " + size + " 字节");
        } catch (Exception e) {
            System.err.println("程序运行出错: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void printLocalHostInfo() throws IOException {
        // 从本机网络配置中解析本地主机地址信息
        InetAddress localHost = InetAddress.getLocalHost();

        System.out.println("本地主机信息");
        System.out.println("主机名: " + localHost.getHostName());
        System.out.println("IP地址: " + localHost.getHostAddress());
    }

    private static void printAllIpsOfCsdn() throws IOException {
        // 对同一域名执行完整DNS解析，返回所有可用IP地址
        InetAddress[] addresses = InetAddress.getAllByName("www.csdn.net");

        System.out.println("\nwww.csdn.net 的所有IP");
        for (int i = 0; i < addresses.length; i++) {
            System.out.println("IP[" + (i + 1) + "]: " + addresses[i].getHostAddress());
        }
    }

    private static long downloadWebPage(String urlString, String outputFile) throws IOException {
        // URL对象负责处理协议、主机与路径等组成部分
        URL url = new URL(urlString);
        // 分块读取，避免一次性加载过大内容
        byte[] buffer = new byte[4096];
        int len;
        // 记录实际写入文件的总字节数
        long totalBytes = 0;

        try (InputStream in = new BufferedInputStream(url.openStream());
             FileOutputStream out = new FileOutputStream(outputFile)) {
            // 循环读取直到流结束（read返回-1）
            while ((len = in.read(buffer)) != -1) {
                out.write(buffer, 0, len);
                totalBytes += len;
            }
        }

        return totalBytes;
    }
}
