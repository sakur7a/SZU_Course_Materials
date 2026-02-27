.ORIG x2000
; 寄存器保存区
ADD R6, R6, #-1
STR R0, R6, #0    
ADD R6, R6, #-1
STR R1, R6, #0    
ADD R6, R6, #-1
STR R2, R6, #0    
ADD R6, R6, #-1
STR R3, R6, #0    
ADD R6, R6, #-1
STR R4, R6, #0    
ADD R6, R6, #-1
STR R7, R6, #0    

; 初始化
LD R4, BUFFER       ; R4 = 缓冲区起始地址
LD R2, MAX_LENGTH   ; R2 = 最大字符数(31)

; 读取输入循环
read_loop:
    LDI R1, KBSR         ; 检查键盘状态
    BRzp read_loop       ; 等待输入
    
    LDI R0, KBDR         ; 读取字符
    
    ; 检查回车键
    LD R3, negEnter      ; R3 = -x000A (xFFF6)
    ADD R3, R0, R3       ; R0 - x000A
    BRz end_read         ; 是回车则结束读取
    
    ; 存储并回显字符
........
    STR R0, R4, #0       ; 存储到缓冲区
    JSR OUTPUT_CHAR      ; 调用输出子程序
    
    ; 更新指针和计数器
    ADD R4, R4, #1       ; 缓冲区指针后移
    ADD R2, R2, #-1      ; 剩余长度减1
    BRp read_loop        ; 如果还有空间继续读取

end_read:
    ; 添加字符串终止符
    AND R3, R3, #0       ; R3 = 0 (x0000)
    STR R3, R4, #0       ; 添加空终止符
    
    ; 打印初始换行分隔输入和输出
    LD R0, enter
    JSR OUTPUT_CHAR
    
    ; 设置打印计数器
    AND R2, R2, #0
    ADD R2, R2, #10      ; R2 = 10（打印10次）

; 打印字符串循环
print_loop:
    LD R4, BUFFER         ; 重置缓冲区指针
    
    ; 打印字符串（直到遇到终止符）
    char_loop:
        LDR R0, R4, #0      ; 读取缓冲区字符
        BRz end_char        ; 遇到空字符结束
        
        JSR OUTPUT_CHAR     ; 输出字符
        ADD R4, R4, #1      ; 指针后移
        BRnzp char_loop
    
    end_char:
        ; 打印字符串后换行
        LD R0, enter
        JSR OUTPUT_CHAR
        
        ; 循环计数
        ADD R2, R2, #-1
        BRp print_loop

; 恢复寄存器并返回
LDR R7, R6, #0      ; 恢复R7
ADD R6, R6, #1
LDR R4, R6, #0      ; 恢复R4
ADD R6, R6, #1
LDR R3, R6, #0      ; 恢复R3
ADD R6, R6, #1
LDR R2, R6, #0      ; 恢复R2
ADD R6, R6, #1
LDR R1, R6, #0      ; 恢复R1
ADD R6, R6, #1
LDR R0, R6, #0      ; 恢复R0
ADD R6, R6, #1
RTI                 ; 从中断返回

; 字符输出子程序
OUTPUT_CHAR          ; 输出R0中的字符（不修改R0）
    ST R1, saveR1    ; 保存R1
    ST R0, saveR0    ; 保存R0
    output_wait:
        LDI R1, DSR      ; 检查显示状态
        BRzp output_wait  ; 等待显示就绪
        STI R0, DDR      ; 输出字符
    
    LD R0, saveR0    ; 恢复R0
    LD R1, saveR1    ; 恢复R1
    RET               ; 返回

; 数据区
BUFFER      .BLKW 32     ; 缓冲区（32字）
MAX_LENGTH  .FILL #31    ; 最大字符数
enter       .FILL x000A  ; 换行符
negEnter    .FILL xFFF6  ; -x000A
KBSR        .FILL xFE00  ; 键盘状态寄存器
KBDR        .FILL xFE02  ; 键盘数据寄存器
DSR         .FILL xFE04  ; 显示状态寄存器
DDR         .FILL xFE06  ; 显示数据寄存器
saveR0      .BLKW 1      ; 临时保存R0
saveR1      .BLKW 1      ; 临时保存R1
.END