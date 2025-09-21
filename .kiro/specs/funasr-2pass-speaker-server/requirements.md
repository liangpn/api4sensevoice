# Requirements Document

## Introduction

基于官方 FunASR 的 `funasr-wss-server-2pass` 进行优化，创建一个支持特定说话人识别的 2pass 语音识别服务器。该项目将创建新的文件而不修改原有代码，包括新的服务器启动脚本、客户端代码和 HTML 界面，以支持特定说话人的语音识别功能。

## Requirements

### Requirement 1: 创建新的服务器组件

**User Story:** 作为开发者，我希望创建新的服务器文件，基于官方的 `funasr-wss-server-2pass` 代码进行扩展，保持原有功能的同时增加说话人验证。

#### Acceptance Criteria

1. WHEN 创建新服务器 THEN 系统 SHALL 创建新的 C++ 服务器文件 `funasr-wss-server-2pass-speaker.cpp`
2. WHEN 创建启动脚本 THEN 系统 SHALL 创建新的启动脚本 `run_server_2pass_speaker.sh` 启动新的说话人验证服务
3. WHEN 运行服务 THEN 系统 SHALL 在不同端口运行新的说话人验证服务，与原服务器并行工作
4. WHEN 保持兼容性 THEN 系统 SHALL 完全兼容原有的 WebSocket 协议和消息格式
5. WHEN 禁用说话人验证 THEN 系统 SHALL 表现与原服务器完全一致

### Requirement 2: 特定说话人验证功能

**User Story:** 作为用户，我希望系统只对特定的单个预注册说话人进行语音识别，其他说话人的音频将被忽略。

#### Acceptance Criteria

1. WHEN 服务器启动 THEN 系统 SHALL 通过参数加载特定说话人的单个 WAV 音频文件
2. WHEN 接收到音频块 THEN 系统 SHALL 在 ASR 处理前进行说话人验证
3. IF 说话人验证通过 THEN 系统 SHALL 继续执行 2pass ASR 处理并返回识别结果
4. IF 说话人验证失败 THEN 系统 SHALL 跳过该音频块的 ASR 处理并返回拒绝消息
5. WHEN 进行说话人验证 THEN 系统 SHALL 仅支持单个预注册说话人

### Requirement 3: 新的客户端支持

**User Story:** 作为开发者，我希望创建新的客户端代码来测试说话人验证功能。

#### Acceptance Criteria

1. WHEN 创建客户端 THEN 系统 SHALL 创建新的 C++ 客户端文件 `funasr-wss-client-2pass-speaker.cpp`
2. WHEN 客户端连接 THEN 系统 SHALL 支持向说话人验证服务器发送音频
3. WHEN 接收响应 THEN 系统 SHALL 正确处理说话人验证成功和失败的响应
4. WHEN 测试功能 THEN 系统 SHALL 支持指定测试音频文件进行验证

### Requirement 4: 新的 Web 界面

**User Story:** 作为用户，我希望通过新的 Web 界面测试说话人验证功能。

#### Acceptance Criteria

1. WHEN 创建界面 THEN 系统 SHALL 创建新的 HTML 文件 `index_speaker.html` 专门适配新的说话人验证服务
2. WHEN 使用界面 THEN 系统 SHALL 支持连接到新的说话人验证服务器（不同端口）
3. WHEN 录音测试 THEN 系统 SHALL 支持实时录音并进行说话人验证
4. WHEN 显示结果 THEN 系统 SHALL 清晰显示说话人验证状态和识别结果
5. WHEN 配置连接 THEN 系统 SHALL 默认连接到新服务的端口地址

### Requirement 5: 参数配置支持

**User Story:** 作为部署人员，我希望通过启动参数配置说话人验证功能。

#### Acceptance Criteria

1. WHEN 启动服务器 THEN 系统 SHALL 支持 `--speaker-file` 参数指定预注册说话人的单个 WAV 音频文件路径
2. WHEN 启动服务器 THEN 系统 SHALL 支持 `--spk-threshold` 参数配置验证阈值（默认值 0.5）
3. WHEN 启动服务器 THEN 系统 SHALL 支持 `--enable-speaker-verification` 参数启用/禁用说话人验证
4. WHEN 参数错误 THEN 系统 SHALL 提供清晰的错误提示并优雅退出

### Requirement 6: 文件组织和部署

**User Story:** 作为部署人员，我希望新创建的文件能够与现有系统和谐共存。

#### Acceptance Criteria

1. WHEN 创建文件 THEN 系统 SHALL 不修改任何现有的原始文件
2. WHEN 部署系统 THEN 系统 SHALL 支持与原有服务器并行运行
3. WHEN 构建项目 THEN 系统 SHALL 提供独立的构建配置
4. WHEN 管理依赖 THEN 系统 SHALL 复用现有的 FunASR 依赖和模型