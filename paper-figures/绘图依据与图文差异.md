# 图2–4内容核对与重绘依据

依据：上传论文正文、Word 内嵌 Office Math 公式、原图2–4。段落号对应同目录 `paper_text.txt`。本文档仅记录核对结果，不更改论文。

## 图2：HSAE-Net总体架构

### 按正文可确定的流程

- FPAN：胸片 I → 图像频域处理（FFT、低/高频掩码、iFFT）→ ResNet-101 → C2、C3、C4、C5 → 各尺度 HFP → 1×1横向卷积与自顶向下上采样相加 → SCSA → 视觉特征 V。C2–C5分辨率依次为原图的1/4、1/8、1/16、1/32（[048]–[064]）。[067]明确在P3、P2融合后使用SCSA。默认选P3，Ratio=(0.18,0.15)（[140]）。
- ASCA：固定医学概念词典 → Tokenize/Embedding → Q及固定Query ID；视觉V → K/Value，并经两层MLP、Softmax、argmax产生Key ID；比较两类ID产生语义掩码；掩码加到点积注意力分数上 → Softmax → 加权视觉响应Z（[085]–[099]）。视觉类别为弱监督预测，不是人工解剖分割。
- CAM：已观察/已生成前缀 y_<t → Embedding → Transformer Context Encoder → Mean Pool → Linear Projection → c_t → 解码器上下文注意力。训练用teacher-forced prefix，推理用generated prefix，均不输入完整参考报告或未来词（[104]）。
- 解码器按式(24)–(28)顺序：Masked Self-Attention → Visual Cross-Attention → Context Attention → FFN → 输出投影与Softmax → y_t；y_t反馈用于下一时刻前缀。

### ASCA精确公式

式(15)：p_j = Softmax(f_cls(v_j)) ∈ R^K。

式(16)：s_j^key = argmax_k p_j[k]。

式(17)：S = Q K^T / sqrt(d) ∈ R^(M×N)。

式(18)：M_sem[i,j] = 0，当 s_i^query = s_j^key；否则为 −∞。

式(19)：S′ = S + M_sem。

式(20)：Z = A V ∈ R^(M×d)，其中 A = softmax(S′)。

### CAM精确公式

式(21)：T_enc = TransformerEncoder(S)。S应是当前文本前缀嵌入。

式(22)：c_raw = (1/L) Σ_i T_enc[i]。

式(23)：c = W_proj · c_raw + b_proj。

式(24)：x_sa = LayerNorm(x + MultiHeadAttention(x,x,x))。

式(25)：x_cross = LayerNorm(x_sa + MultiHeadAttention(x_sa,V,V))。

式(26)：c_expanded = c · 1_L^T。此为原文写法，语义是沿序列长度广播；按行向量序列实现时转置记法需相应调整。

式(27)：x′ = x_cross + MultiHeadAttention_ctx(x_cross,c_expanded,c_expanded)。即Q来自解码隐藏状态，K/V来自上下文。

式(28)：x_output = FFN(LayerNorm(x′)) + x′。

## 图3：SCSA

按原图细节：SMSA中，F沿H/W两个方向分别AvgPool → Split into 4 channel groups → 共享多尺度DWConv1D（核3、5、7、9）→ 每个方向分别Concat、GroupNorm(4 groups)、Sigmoid → 两个方向权重与F逐元素相乘，形成F_s。

F_s进入PCSA：AvgPool/Downsample → GroupNorm(1 group) → 三条1×1 Conv形成Q、K、V → Channel-wise Single-Head Self-Attention → 空间AvgPool → Sigmoid → 与F_s逐元素相乘，输出F′。

原图三个投影分支误标为V/K/V，应改为Q/K/V。

正文式(10)：z_h = (1/W)Σ_j F[:,:,j]，z_w = (1/H)Σ_i F[:,i,:]。

正文式(11)：w_s = σ(GroupNorm(Conv1D_multi(z_h) + Conv1D_multi(z_w)))。

正文式(12)：F_down = DownSample(F)；Q,K,V = Linear_Q(F_down), Linear_K(F_down), Linear_V(F_down)。

正文式(13)：A = Softmax(QK^T/sqrt(d))；w_c = σ(Mean_(H′,W′)(AV))。

正文式(14)：F′ = F ⊙ w_s ⊙ w_c。

注意：正文式(11)把两个方向写成相加再归一化，原图是两个方向独立归一化后相乘；式(12)简写为从F下采样，而原图PCSA实际接收SMSA输出。重绘采用原图的模块细节时应在说明中注明这一差异，不能称图文完全一致。

## 图4：HFP

### 精确公式

式(5)：F_i = IFFT(M_high ⊙ FFT(C_i))。

式(6)：u_SP = σ(Conv_spatial(F_i))；u_CP = σ(Conv_channel(F_i))。

式(7)：C_i^HFP = Conv_(3×3)[(u_CP ⊙ C_i) + (u_SP ⊙ C_i)]。

原始C_i必须分别连到通道和空间两个逐元素乘法；两个加权分支随后相加，再接3×3 Conv。不能画成两种注意力串行相乘。

### 池化与激活顺序

原图通道分支明确为F_i → 两条GAP/GMP分支 → 每支σ → 每支1×1 Conv → Concat → 1×1 Conv → Sigmoid → 与C_i相乘。原图空间分支为F_i → 1×1 Conv → Sigmoid → 与C_i相乘。

正文式(6)未展开GAP/GMP/Concat，也没有规定其与中间激活的相对顺序。因此不能从式(6)推断应将池化移到Conv或Sigmoid之后；保留原图的详细池化顺序是可说明的绘图选择。

原图iFFT后的“dct”与FFT方法冲突，应改为“F_i / High-frequency features”。原图输出P_i^HFP与正文式(7)的C_i^HFP不同；C用于骨干与HFP特征，P用于之后金字塔输出，按正文统一为C_i^HFP更清楚。

## 交付给用户时值得说明的实质差异

1. 频域处理位置：原图在ResNet之后，[048][054]在ResNet之前；图2按正文前置。
2. CAM方向与顺序：[038]称上下文作Q，[113]称FFN后使用CAM，但式(27)(28)规定隐藏状态作Q、上下文作K/V，且CAM在FFN前；图2按明确公式绘制。
3. SCSA方向融合：正文式(11)与原图不一致；图3保留原图两个方向分别产生权重的结构，并修正Q/K/V文字。
4. ASCA正文称“软语义约束”，式(18)实际采用0/−∞硬掩码；图中用中性的“Semantic mask”，不增加软约束声明。
5. 原图CMN/CCA没有在当前方法正文中得到充分定义，不能把CMN当成正文确认的必需组件。可按FPAN、ASCA、CAM及Transformer编码器—解码器统一命名；省略未解释的CMN时宜说明依据是当前正文。

## 不应从图中新增的实现结论

- [133]给出视觉输出维度2048、Transformer隐层512，但未明确P3通道数和投影所在位置。图2不必标“P3:2048→512”，避免把全局超参数变成未经证实的内部形状。
- 式(3)为M_high=1−M_low，式(4)为I_Enhanced=IFFT(F⊙M_low)+IFFT(F⊙M_high)。严格按互补掩码直接相加会还原输入，未单独解释增强来自何种参数；不能为“修正”论文而自行增加可学习系数、滤波公式或门控机制。
- 原稿没有提供足够信息规定ASCA某个Query无同类视觉Key时的处理；绘图不添加未经论证的fallback。
