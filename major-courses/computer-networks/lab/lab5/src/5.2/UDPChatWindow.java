import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

// 实验5.2：基于UDP的数据报聊天窗口
public class UDPChatWindow extends JFrame {
    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss");

    private final int localPort;
    private final InetAddress remoteAddress;
    private final int remotePort;
    private final String userName;

    private DatagramSocket socket;
    private Thread receiverThread;

    private final JTextArea historyArea = new JTextArea();
    private final JTextField inputField = new JTextField();

    public UDPChatWindow(int localPort, InetAddress remoteAddress, int remotePort, String userName) {
        this.localPort = localPort;
        this.remoteAddress = remoteAddress;
        this.remotePort = remotePort;
        this.userName = userName;

        setTitle("UDP聊天程序 - " + userName);
        setSize(700, 500);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);

        // 按顺序完成界面初始化、事件绑定和网络启动
        initUi();
        bindEvents();
        startUdp();
    }

    private void initUi() {
        // 上中下三段式布局：连接信息、聊天记录、输入区
        setLayout(new BorderLayout(8, 8));

        JPanel topPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        topPanel.add(new JLabel("本地端口: " + localPort));
        topPanel.add(new JLabel("远端: " + remoteAddress.getHostAddress() + ":" + remotePort));

        historyArea.setEditable(false);
        historyArea.setLineWrap(true);
        historyArea.setWrapStyleWord(true);

        JScrollPane scrollPane = new JScrollPane(historyArea);
        scrollPane.setBorder(BorderFactory.createTitledBorder("聊天内容"));

        JPanel bottomPanel = new JPanel(new BorderLayout(8, 8));
        inputField.setBorder(BorderFactory.createTitledBorder("输入文本"));

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        JButton sendButton = new JButton("确定");
        JButton clearButton = new JButton("清空");
        JButton exitButton = new JButton("退出");

        buttonPanel.add(sendButton);
        buttonPanel.add(clearButton);
        buttonPanel.add(exitButton);

        sendButton.addActionListener(e -> sendCurrentText());
        clearButton.addActionListener(e -> inputField.setText(""));
        exitButton.addActionListener(e -> closeAndExit());
        inputField.addActionListener(e -> sendCurrentText());

        bottomPanel.add(inputField, BorderLayout.CENTER);
        bottomPanel.add(buttonPanel, BorderLayout.EAST);

        add(topPanel, BorderLayout.NORTH);
        add(scrollPane, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
    }

    private void bindEvents() {
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                closeAndExit();
            }
        });
    }

    private void startUdp() {
        try {
            // 绑定本地端口用于收发UDP数据报
            socket = new DatagramSocket(localPort);
            appendHistory("系统", "UDP监听已启动，端口 " + localPort);
        } catch (IOException e) {
            JOptionPane.showMessageDialog(this, "启动UDP失败: " + e.getMessage(), "错误", JOptionPane.ERROR_MESSAGE);
            dispose();
            return;
        }

        receiverThread = new Thread(() -> {
            // 后台循环接收，直到socket关闭
            byte[] buffer = new byte[2048];
            while (!socket.isClosed()) {
                DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
                try {
                    socket.receive(packet);
                    // 仅按本次数据报有效长度解码
                    String msg = new String(packet.getData(), packet.getOffset(), packet.getLength(), StandardCharsets.UTF_8);
                    String from = packet.getAddress().getHostAddress() + ":" + packet.getPort();
                    System.out.println("接收消息(" + from + "): " + msg);
                    appendHistory("对方", msg);
                } catch (IOException e) {
                    if (!socket.isClosed()) {
                        appendHistory("系统", "接收异常: " + e.getMessage());
                    }
                }
            }
        }, "udp-receiver");
        receiverThread.setDaemon(true);
        receiverThread.start();
    }

    private void sendCurrentText() {
        String text = inputField.getText().trim();
        if (text.isEmpty()) {
            return;
        }

        // 将文本编码后发往目标地址和端口
        byte[] data = text.getBytes(StandardCharsets.UTF_8);
        DatagramPacket packet = new DatagramPacket(data, data.length, remoteAddress, remotePort);
        try {
            socket.send(packet);
            System.out.println("发送消息(" + remoteAddress.getHostAddress() + ":" + remotePort + "): " + text);
            appendHistory(userName, text);
            inputField.setText("");
        } catch (IOException e) {
            appendHistory("系统", "发送失败: " + e.getMessage());
        }
    }

    private void appendHistory(String sender, String message) {
        String line = "[" + LocalDateTime.now().format(FORMATTER) + "] " + sender + ": " + message + System.lineSeparator();
        // 通过事件派发线程更新Swing组件，避免线程安全问题
        SwingUtilities.invokeLater(() -> historyArea.append(line));
    }

    private void closeAndExit() {
        // 关闭socket会使接收线程从阻塞中退出
        if (socket != null && !socket.isClosed()) {
            socket.close();
        }
        dispose();
        System.exit(0);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println("用法: java UDPChatWindow <localPort> <remoteHost> <remotePort> [userName]");
            System.out.println("示例1: java UDPChatWindow 6000 127.0.0.1 6001 A");
            System.out.println("示例2: java UDPChatWindow 6001 127.0.0.1 6000 B");
            return;
        }

        int localPort = Integer.parseInt(args[0]);
        InetAddress remoteHost = InetAddress.getByName(args[1]);
        int remotePort = Integer.parseInt(args[2]);
        String userName = args.length >= 4 ? args[3] : ("用户-" + localPort);

        // 在EDT中创建并显示窗口
        SwingUtilities.invokeLater(() -> {
            UDPChatWindow window = new UDPChatWindow(localPort, remoteHost, remotePort, userName);
            window.setVisible(true);
        });
    }
}
