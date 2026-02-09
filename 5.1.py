import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow.keras.preprocessing.image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ===================== 1. 配置全局参数 =====================
# 基础参数
IMG_SIZE = 28  # MNIST数据集的标准尺寸是28x28像素
NUM_CLASSES = 10  # 分类数：对应数字0-9共10个类别，输出层神经元数量
BATCH_SIZE = 64  # 批大小：每次训练使用的样本数，影响训练速度、内存占用和梯度更新的稳定性
EPOCHS = 20  # 训练轮数：完整遍历训练集的次数，增加轮数有助于模型充分学习，但可能过拟合
LEARNING_RATE = 0.001  # 学习率：Adam优化器的初始步长，控制参数更新幅度
SEED = 42  # 随机种子：确保每次运行的随机初始化、数据打乱等操作结果一致，保证可复现性

# 路径配置（需要根据用户实际情况修改）
CUSTOM_IMAGE_PATH = "5.jpg"  # 自定义手写数字图片的路径
SAVE_MODEL_PATH = "mnist_cnn_model.keras"  # 训练好的模型保存路径，Keras格式

# 设置随机种子保证可复现性
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei"]  # 指定中文字体为黑体，解决绘图时中文乱码
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

# ===================== 2. 数据准备与预处理 =====================
def load_and_preprocess_data():
    """
    加载MNIST数据集并完成预处理。
    返回：预处理后的训练集、验证集和测试集数据及标签。
    """
    # 加载MNIST数据集（TensorFlow内置数据集）
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # MNIST包含60000个训练样本（x_train, y_train）和10000个测试样本（x_test, y_test）

    # 数据探索：打印数据集基本信息，帮助理解数据分布
    print("=" * 50)
    print("数据探索结果：")
    print(f"训练集形状: {x_train.shape}, 测试集形状: {x_test.shape}")  # (60000, 28, 28) 和 (10000, 28, 28)
    print(f"像素值范围: {x_train.min()} ~ {x_train.max()}")  # 原始像素值为0（黑）到255（白）
    # 使用np.bincount统计每个标签的出现次数，观察类别是否均衡
    print(f"训练集标签分布: {np.bincount(y_train)}")
    print(f"测试集标签分布: {np.bincount(y_test)}")

    # 预处理1：归一化（将像素值从0-255缩放到0-1之间）
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # 归一化可以加速模型收敛，防止梯度计算中出现过大或过小的数值

    # 预处理2：添加通道维度（适配CNN输入要求）
    # CNN期望输入形状为（高度，宽度，通道数）。灰度图通道数为1。
    x_train = np.expand_dims(x_train, axis=-1)  # 从 (60000, 28, 28) 变为 (60000, 28, 28, 1)
    x_test = np.expand_dims(x_test, axis=-1)  # 从 (10000, 28, 28) 变为 (10000, 28, 28, 1)

    # 预处理3：标签独热编码（One-hot Encoding）
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)


    # 划分验证集：从训练集中拆分15%作为验证集，用于在训练时监控模型泛化能力
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=0.15,  # 验证集占比15%
        random_state=SEED,  # 固定随机状态，确保每次划分一致
        stratify=y_train  # 分层抽样：保持验证集和训练集中各类别比例与原训练集一致
    )

    # 打印数据集划分结果
    print("\n数据集划分结果：")
    print(f"训练集: {x_train.shape}, 验证集: {x_val.shape}, 测试集: {x_test.shape}")
    print("=" * 50)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ===================== 3. 模型设计与训练 =====================
def build_cnn_model(use_dropout=True):
    """
    构建卷积神经网络（CNN）模型。
    参数：
        use_dropout (bool): 是否在展平层后添加Dropout层以防止过拟合。
    返回：
        编译好的Keras模型对象。
    """
    model = models.Sequential([  # 顺序模型：层按定义的顺序堆叠执行
        # 第一卷积块：提取图像的低级特征（如边缘、角点）
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        # 使用32个3x3的卷积核，激活函数为ReLU。输入形状为(28, 28, 1)。
        layers.MaxPooling2D((2, 2)),  # 2x2最大池化层，将特征图尺寸减半，增强特征的空间不变性

        # 第二卷积块：提取中级特征（如线条、曲线）
        layers.Conv2D(64, (3, 3), activation='relu'),  # 增加至64个卷积核
        layers.MaxPooling2D((2, 2)),  # 再次池化，进一步降低维度

        # 第三卷积块：提取高级特征（如形状、部件组合）
        layers.Conv2D(128, (3, 3), activation='relu'),  # 增加至128个卷积核，增加模型容量
        # 注意：此处未接池化层，是为了在降低维度后仍保留足够丰富的特征信息用于分类

        # 展平层：将三维的特征图（height, width, channels）转换为一维向量，作为全连接层的输入
        layers.Flatten(),

        # Dropout层：防止过拟合的关键技术。训练时随机“丢弃”30%的神经元，迫使网络学习更鲁棒的特征
        layers.Dropout(0.3) if use_dropout else layers.Layer(),  # 使用条件表达式决定是否添加该层

        # 全连接层（Dense层）：整合卷积层提取的所有特征，进行非线性组合和决策
        layers.Dense(128, activation='relu'),  # 128个神经元，ReLU激活

        # 输出层：对应10个数字类别，使用Softmax激活函数输出每个类别的概率（总和为1）
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # 编译模型：配置模型的优化器、损失函数和评估指标
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)  # Adam优化器，自适应调整学习率
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',  # 分类交叉熵损失函数，适用于多分类任务
        metrics=['accuracy']  # 评估指标为准确率
    )

    # 打印模型的结构摘要，包括每层的输出形状和参数数量
    model.summary()

    return model


def train_model(model, x_train, y_train, x_val, y_val, use_data_augmentation=True):
    """
    使用给定数据训练CNN模型。
    参数：
        model: 编译好的Keras模型。
        x_train, y_train: 训练数据及标签。
        x_val, y_val: 验证数据及标签。
        use_data_augmentation (bool): 是否使用数据增强来扩充训练集。
    返回：
        model: 训练好的模型。
        history: 包含训练过程中各指标历史记录的对象。
    """
    # 定义早停策略（Early Stopping）回调函数
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # 监控验证集损失
        patience=5,  # 容忍轮数：若验证损失连续5轮不再下降，则停止训练
        restore_best_weights=True  # 恢复训练过程中验证损失最低时的模型权重
    )
    # 数据增强（Data Augmentation）：通过对训练图像进行随机变换，人工增加数据多样性，提升模型泛化能力
    if use_data_augmentation:
        from keras.src.legacy.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=15,  # 随机旋转角度范围（-15度到+15度），模拟手写倾斜
            width_shift_range=0.15,  # 随机水平平移范围（±15%宽度），模拟位置变化
            height_shift_range=0.15,  # 随机垂直平移范围（±15%高度）
            zoom_range=0.15,  # 随机缩放范围（85%到115%），模拟大小变化
            shear_range=0.1,  # 随机剪切变换范围（-0.1到+0.1弧度），模拟笔迹扭曲
            fill_mode='constant',  # 填充模式：用常数填充变换后产生的空白区域
            cval=1.0  # 填充常数值为1.0（白色），以匹配MNIST的白色数字背景
        )
        datagen.fit(x_train)  # 计算数据增强所需的统计信息（如均值、标准差）

        # 使用数据增强生成器进行模型训练
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),  # 生成增强后的数据流
            epochs=EPOCHS,  # 最大训练轮数
            validation_data=(x_val, y_val),  # 每轮结束后用验证集评估
            callbacks=[early_stopping]  # 应用早停回调
        )
    else:
        # 不使用数据增强的标准训练流程
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping]
        )

    # 训练完成后，将模型保存到指定路径
    model.save(SAVE_MODEL_PATH)
    print(f"\n模型已保存至: {SAVE_MODEL_PATH}")

    return model, history


# ===================== 4. 模型性能分析 =====================
def analyze_model_performance(model, x_test, y_test, history):
    """
    对训练好的模型进行全面性能分析。
    包括：绘制学习曲线、在测试集上评估、生成混淆矩阵和分类报告。
    """
    # 1. 绘制训练/验证损失和准确率曲线
    plt.figure(figsize=(12, 4))  # 创建宽12英寸、高4英寸的图形

    # 子图1：损失曲线
    plt.subplot(1, 2, 1)  # 1行2列布局中的第1个子图
    plt.plot(history.history['loss'], label='训练损失')  # 训练集损失变化
    plt.plot(history.history['val_loss'], label='验证损失')  # 验证集损失变化
    plt.title('训练/验证损失曲线')  # 标题
    plt.xlabel('Epoch')  # X轴：训练轮次
    plt.ylabel('Loss')  # Y轴：损失值
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格，便于观察

    # 子图2：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('训练/验证准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # 自动调整子图间距，避免重叠
    plt.savefig('training_curves.png')  # 将图形保存为PNG文件
    plt.show()  # 显示图形

    # 2. 在独立的测试集上评估模型的最终性能
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)  # `verbose=0`不显示评估过程
    print("\n" + "=" * 50)
    print(f"测试集准确率: {test_acc:.4f}")  # 格式化输出，保留4位小数
    print(f"测试集损失: {test_loss:.4f}")
    print("=" * 50)

    # 3. 生成并可视化混淆矩阵（Confusion Matrix）
    # 获取模型对测试集所有样本的预测结果
    y_pred = model.predict(x_test)  # 输出为概率分布，形状 (10000, 10)
    y_pred_classes = np.argmax(y_pred, axis=1)  # 将概率转换为类别预测（取最大概率的索引）
    y_true_classes = np.argmax(y_test, axis=1)  # 将one-hot标签转换回原始类别标签

    # 计算混淆矩阵：行代表真实类别，列代表预测类别
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    # 使用Seaborn绘制热力图（Heatmap）直观展示混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,  # 在每个单元格中显示数值
        fmt='d',  # 数值格式为整数（decimal）
        cmap='Blues',  # 使用蓝色系颜色映射
        xticklabels=range(NUM_CLASSES),  # X轴刻度标签为0-9
        yticklabels=range(NUM_CLASSES)  # Y轴刻度标签为0-9
    )
    plt.title('混淆矩阵')  # 标题
    plt.xlabel('预测标签')  # X轴标签
    plt.ylabel('真实标签')  # Y轴标签
    plt.savefig('confusion_matrix.png')  # 保存混淆矩阵图
    plt.show()

    # 4. 打印详细的分类报告（Precision, Recall, F1-Score）
    print("\n分类报告：")
    print(classification_report(y_true_classes, y_pred_classes))
    # - Precision（精确率）：预测为正的样本中，真实为正的比例。衡量预测的准确性。
    # - Recall（召回率）：真实为正的样本中，被预测为正的比例。衡量预测的覆盖率。
    # - F1-score（F1分数）：精确率和召回率的调和平均数，综合衡量模型性能。
    # - Support（支持数）：每个类别的真实样本数量。


# ===================== 5. 自定义图片预测（核心优化） =====================
def preprocess_custom_image(image_path):
    """
    对用户提供的自定义手写数字图片进行预处理，使其格式与MNIST训练数据一致。
    这是提升预测准确率的关键步骤！
    参数：
        image_path (str): 自定义图片的文件路径。
    返回：
        original_img (PIL.Image): 原始图片对象（用于显示）。
        processed_img (numpy.ndarray): 预处理后的图片数组，形状为(1,28,28,1)，可直接输入模型。
    """
    # 1. 加载图片并强制转换为灰度图（'L'模式）
    img = Image.open(image_path).convert('L')
    # 无论原图是彩色还是灰度，都转为单通道灰度图，与MNIST一致。

    # 2. 调整尺寸为28x28像素（双线性插值法）
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    # 必须调整为模型规定的输入尺寸。

    # 3. 将PIL图像对象转换为NumPy数组，并将像素值从0-255归一化到0-1的浮点数
    img_array = np.array(img).astype('float32') / 255.0

    # 4. 反转颜色（核心步骤，适配MNIST格式）
    img_array = 1.0 - img_array
    # MNIST数据集是“黑底白字”（背景接近0，数字接近1）。
    # 普通图片通常是“白底黑字”，所以需要反转。如果原图已是黑底白字，此步骤可能会降低准确性。

    # 5. 核心优化：二值化（阈值处理）
    img_array = np.where(img_array > 0.5, 1.0, 0.0)
    # 将归一化后的灰度图转换为只有0和1的绝对黑白图像。
    # 阈值0.5是一个经验值。此操作可以去除手写数字的模糊边缘和灰色噪声，使特征更清晰，与MNIST风格更接近。

    # 6. 添加维度以适配CNN模型输入格式
    img_array = np.expand_dims(img_array, axis=-1)  # 添加通道维度：(28,28) -> (28,28,1)
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度：(28,28,1) -> (1,28,28,1)
    # 模型预测通常以批次为单位，即使单张图片也需要保持4维形状。

    return img, img_array  # 返回原始图像（用于显示）和处理后的数组（用于预测）


def predict_custom_image(model, image_path):
    """
    使用训练好的模型对单张自定义图片进行数字识别。
    参数：
        model: 训练好的Keras模型。
        image_path (str): 待预测图片的路径。
    返回：
        predicted_class (int): 预测的数字（0-9）。
        confidence (float): 预测置信度（百分比）。
    """
    # 调用预处理函数处理图片
    original_img, processed_img = preprocess_custom_image(image_path)

    # 模型预测：输入形状为(1,28,28,1)，输出形状为(1,10)，即10个类别的概率分布
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]  # 找出概率最大的类别索引
    confidence = np.max(prediction) * 100  # 最高概率值，转换为百分比作为置信度

    # 可视化预测结果
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img, cmap='gray')  # 显示原始灰度图
    plt.title(f"预测结果: {predicted_class} (置信度: {confidence:.2f}%)", fontsize=14)
    plt.axis('off')  # 隐藏坐标轴
    plt.savefig('custom_prediction.png')  # 保存预测结果图
    plt.show()

    # 在控制台打印预测结果
    print("\n自定义图片预测结果：")
    print(f"预测数字: {predicted_class}")
    print(f"置信度: {confidence:.2f}%")

    return predicted_class, confidence


# ===================== 主函数（程序执行入口） =====================
if __name__ == "__main__":
    # 1. 数据准备与预处理
    print("开始数据准备与预处理...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    # 2. 构建CNN模型
    print("\n开始构建CNN模型...")
    model = build_cnn_model(use_dropout=True)  # 启用Dropout以防止过拟合

    # 3. 训练模型（启用数据增强提升泛化能力）
    print("\n开始训练模型...")
    model, history = train_model(
        model, x_train, y_train, x_val, y_val,
        use_data_augmentation=True  # 启用数据增强
    )

    # 4. 模型性能分析
    print("\n开始模型性能分析...")
    analyze_model_performance(model, x_test, y_test, history)

    # 5. 自定义图片预测（用户测试）
    print("\n开始自定义图片预测...")
    # 检查用户指定的自定义图片文件是否存在
    if os.path.exists(CUSTOM_IMAGE_PATH):
        # 如果存在，调用预测函数
        predict_custom_image(model, CUSTOM_IMAGE_PATH)
    else:
        # 如果不存在，给出详细的排查指引
        print(f"\n警告：未找到自定义图片 {CUSTOM_IMAGE_PATH}")
        print("请检查：")
        print("1. 图片路径/文件名是否正确（注意扩展名，如：5.jpg 与 5.png 不同）")
        print("2. 图片是否与.py代码文件在同一目录下")
        print("3. 建议：使用28×28像素、背景纯净、笔画清晰的纯黑白手写数字图，避免模糊或JPEG压缩")