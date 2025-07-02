#include <cstdint>
#include <stdio.h>
#include <modbus.h>

int main(void) {
    modbus_t* mb;

    // 创建 Modbus TCP 连接
    mb = modbus_new_tcp("192.168.1.199", 502);
    if (modbus_connect(mb) == -1) {
        fprintf(stderr, "Connection failed: %s\n", modbus_strerror(errno));
        modbus_free(mb);
        return -1;
    }

    // 定义多个 IO 口的状态数据
    struct IOData {
        uint16_t address;       // 寄存器地址
        uint16_t values[2];     // 要写入的寄存器值
    };



    IOData io_states[] = {
        {0x20, {0x0001, 0x0000}}, // IO 口 0 的亮状态

        {0x20, {0x0002, 0x0000}}, // IO 口 1 的亮状态

        {0x20, {0x0004, 0x0000}}  // IO 口 2 的亮状态

    };

    // 循环写入每个 IO 口的状态
    for (const auto& io : io_states) {
        int writeResult = modbus_write_registers(mb, io.address, sizeof(io.values) / sizeof(io.values[0]), io.values);
        if (writeResult == -1) {
            fprintf(stderr, "Write failed for address %d: %s\n", io.address, modbus_strerror(errno));
        }
        else {
            printf("Write successful for address %d: %d registers written.\n", io.address, writeResult);
        }
    }

    // 关闭连接并释放资源
    modbus_close(mb);
    modbus_free(mb);

    return 0;
}