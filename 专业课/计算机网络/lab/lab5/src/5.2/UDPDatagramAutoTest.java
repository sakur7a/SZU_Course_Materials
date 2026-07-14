import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;

// 实验5.2：UDP数据报自动互发测试（A/B两个端口）
public class UDPDatagramAutoTest {
    public static void main(String[] args) throws Exception {
        // 在同一台机器上模拟两个UDP端点
        int portA = 6100;
        int portB = 6101;

        try (DatagramSocket socketA = new DatagramSocket(portA);
             DatagramSocket socketB = new DatagramSocket(portB)) {

            System.out.println("UDP自动测试启动: A=" + portA + ", B=" + portB);

            // A -> B 发送测试消息
            byte[] sendData = "你好啊".getBytes(StandardCharsets.UTF_8);
            DatagramPacket sendPacket = new DatagramPacket(sendData, sendData.length, InetAddress.getByName("127.0.0.1"), portB);
            socketA.send(sendPacket);
            System.out.println("A发送 -> B: 你好啊");

            // B 接收并解码
            byte[] recvBuffer = new byte[1024];
            DatagramPacket recvPacket = new DatagramPacket(recvBuffer, recvBuffer.length);
            socketB.receive(recvPacket);
            String recvMsg = new String(recvPacket.getData(), recvPacket.getOffset(), recvPacket.getLength(), StandardCharsets.UTF_8);
            System.out.println("B接收 <- A: " + recvMsg);

            // B -> A 回执
            byte[] replyData = "收到".getBytes(StandardCharsets.UTF_8);
            DatagramPacket replyPacket = new DatagramPacket(replyData, replyData.length, InetAddress.getByName("127.0.0.1"), portA);
            socketB.send(replyPacket);
            System.out.println("B发送 -> A: 收到");

            // A 接收回执
            DatagramPacket recvPacketA = new DatagramPacket(recvBuffer, recvBuffer.length);
            socketA.receive(recvPacketA);
            String replyMsg = new String(recvPacketA.getData(), recvPacketA.getOffset(), recvPacketA.getLength(), StandardCharsets.UTF_8);
            System.out.println("A接收 <- B: " + replyMsg);

            System.out.println("UDP自动测试通过");
        }
    }
}
