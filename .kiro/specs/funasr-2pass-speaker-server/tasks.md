# Implementation Plan

- [x] 1. 创建说话人验证服务器核心组件
  - 基于原有 `funasr-wss-server-2pass.cpp` 创建新的服务器文件 `funasr-wss-server-2pass-speaker.cpp`
  - 集成说话人验证模块到现有的 WebSocket 处理流程中
  - 添加新的命令行参数支持（--speaker-file, --sv-model-dir, --spk-threshold, --enable-speaker-verification）
  - 保持与原有服务器完全兼容的 WebSocket 协议和消息格式
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.1, 5.2, 5.3, 5.4_

- [x] 2. 实现说话人验证核心算法
  - 创建 SpeakerVerifier 类，支持说话人特征提取和验证
  - 集成 FunASR 的 `damo/speech_campplus_sv_zh-cn_16k-common` 模型进行特征提取
  - 实现多人音频分割功能，复用现有 VAD API (`FsmnVadInferBuffer`)
  - 实现音频段说话人验证和过滤拼接功能
  - 实现余弦相似度计算和阈值判断逻辑
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3. 创建新的启动脚本和配置
  - 基于 `run_server_2pass.sh` 创建新的启动脚本 `run_server_2pass_speaker.sh`
  - 配置新的端口（10096）避免与原服务器冲突
  - 添加说话人验证模型路径配置
  - 添加说话人参考文件和阈值参数配置
  - 确保新服务器可以与原服务器并行运行
  - _Requirements: 1.2, 1.5, 5.1, 5.2, 5.3, 6.2_

- [x] 4. 创建新的客户端测试工具
  - 基于 `funasr-wss-client-2pass.cpp` 创建新的客户端 `funasr-wss-client-2pass-speaker.cpp`
  - 支持连接到说话人验证服务器（新端口）
  - 处理扩展的 JSON 响应格式（speaker_verified, speaker_confidence 字段）
  - 支持测试不同说话人的音频文件进行验证
  - 添加说话人验证状态的显示和日志输出
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 5. 创建新的 Web 界面
  - 基于 `samples/html/static/index.html` 创建新的界面 `index_speaker.html`
  - 修改默认连接地址为新的说话人验证服务器端口（10096）
  - 添加说话人验证状态显示区域（验证状态、置信度）
  - 扩展 JavaScript 处理逻辑，支持新的响应字段
  - 优化界面布局，清晰显示说话人验证结果和 ASR 识别结果
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 6. 集成和构建配置
  - 更新 `FunASR/runtime/websocket/CMakeLists.txt` 添加新的可执行文件 `funasr-wss-server-2pass-speaker` 和 `funasr-wss-client-2pass-speaker`
  - 在 CMakeLists.txt 中添加说话人验证模型的依赖库链接
  - 更新 `samples/cpp/websocket_client/CMakeLists.txt` 支持编译新的客户端
  - 验证新的可执行文件能够正确编译并链接所需的 FunASR 库
  - 确保构建过程不影响原有的 `funasr-wss-server-2pass` 和 `funasr-wss-client-2pass` 的编译
  - _Requirements: 6.1, 6.3, 6.4_