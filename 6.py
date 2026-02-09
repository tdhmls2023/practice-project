import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os
import seaborn as sns

# 评估指标：混淆矩阵、分类报告（精确率、召回率、F1）
from sklearn.metrics import confusion_matrix, classification_report

# PyTorch神经网络模块：定义层、激活函数、损失函数
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 优化器（Adam）
from torch.optim.lr_scheduler import CosineAnnealingLR  # 学习率调度器
from torchsummary import summary  # 模型结构可视化（输出层维度、参数）

import warnings

warnings.filterwarnings('ignore')  # 忽略无关警告（如数据加载提示）

# 设置Matplotlib中文显示（解决中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 自动选择计算设备：优先使用GPU（CUDA），无则用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")  # 打印当前使用的设备（便于调试）


class DataProcessor:

    def __init__(self, batch_size=64):
        """
        初始化参数
        :param batch_size: 每次迭代加载的样本数（批量大小），默认64
        """
        self.batch_size = batch_size  # 批量大小（影响训练效率和稳定性）
        # 类别名称（中文标注，便于分析）：对应Fashion-MNIST的10个服装类别
        self.classes = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
                        '凉鞋', '衬衫', '运动鞋', '包', '短靴']

        # 图像预处理流水线：将PIL图片→张量→标准化
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将PIL图片转换为张量（范围0-1）
            transforms.Normalize((0.5,), (0.5,))  # 标准化：(x-0.5)/0.5 → 范围-1到1
        ])

        # 初始化数据加载器和数据集（后续load_data方法赋值）
        self.trainloader = None  # 训练集加载器
        self.testloader = None  # 测试集加载器
        self.trainset = None  # 训练集（原始数据）
        self.testset = None  # 测试集（原始数据）

    def load_data(self):
        """
        加载Fashion-MNIST数据集并创建数据加载器
        :return: trainloader, testloader - 训练/测试集加载器
        """
        # 加载训练集：root=数据保存路径，train=True=训练集，download=True=自动下载
        self.trainset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=self.transform)
        # 加载测试集：train=False=测试集
        self.testset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=self.transform)

        # 创建训练集加载器：shuffle=True=打乱数据（避免过拟合），num_workers=0=单进程（Windows兼容）
        self.trainloader = DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # 创建测试集加载器：shuffle=False=不打乱（评估时无需打乱）
        self.testloader = DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        return self.trainloader, self.testloader

    def visualize_random_samples(self):
        """
        随机展示每个类别的样本图像（生成可视化图片保存）
        """
        # 创建2行5列的子图（10个类别），设置画布大小
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        fig.suptitle('Fashion-MNIST 每个类别随机样本', fontsize=16)  # 总标题

        # 遍历10个类别
        for i in range(10):
            # 找到当前类别的所有样本索引
            class_indices = np.where(self.trainset.targets.numpy() == i)[0]
            # 随机选择一个样本索引（避免固定展示第一个样本）
            random_idx = np.random.choice(class_indices)
            img, label = self.trainset[random_idx]  # 获取样本图片和标签

            # 反归一化：将标准化后的张量（-1~1）转回0~1，便于可视化
            img = img.squeeze().numpy() * 0.5 + 0.5

            # 选择对应的子图（i//5=行索引，i%5=列索引）
            ax = axes[i // 5, i % 5]
            ax.imshow(img, cmap='gray')  # 以灰度图显示
            ax.set_title(f'类别 {i}: {self.classes[label]}', fontsize=12)  # 子图标题（类别+名称）
            ax.axis('off')  # 关闭坐标轴（更美观）

        plt.tight_layout()  # 自动调整子图间距
        # 保存图片：dpi=150=高清，bbox_inches='tight'=裁剪多余空白
        plt.savefig('random_class_samples.png', dpi=150, bbox_inches='tight')
        plt.close()  # 关闭画布（释放内存）
        print("✅ 已保存每个类别的随机样本图像: random_class_samples.png")


class FashionCNN(nn.Module):
    """
    卷积神经网络（CNN）模型类：用于服装图像分类
    结构：3个卷积块（卷积+批归一化+激活+池化） + 3个全连接层
    输入：1×28×28灰度图 → 输出：10个类别的概率
    """

    def __init__(self):
        super(FashionCNN, self).__init__()  # 继承nn.Module的初始化

        # 卷积块1：输入1×28×28 → 输出32×14×14
        self.conv_block1 = nn.Sequential(
            # 卷积层：in_channels=1（输入通道），out_channels=32（输出通道/卷积核数），kernel_size=3（卷积核大小），padding=1（边缘填充1圈0，保持尺寸）
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # 批量归一化：加速训练，防止梯度消失
            nn.ReLU(inplace=True),  # ReLU激活函数：引入非线性，inplace=True=原地运算（节省内存）
            nn.MaxPool2d(2, 2)  # 最大池化：核大小2×2，步长2 → 尺寸减半（28→14）
        )

        # 卷积块2：输入32×14×14 → 输出64×7×7
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 输入32通道→输出64通道
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 尺寸减半（14→7）
        )

        # 卷积块3：输入64×7×7 → 输出128×3×3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输入64通道→输出128通道
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 尺寸减半（7→3，向下取整）
        )

        # 全连接层：输入128×3×3（展平后） → 输出10（类别数）
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),  # 展平：128×3×3=1152 → 256维
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Dropout：随机失活30%神经元（防止过拟合）
            nn.Linear(256, 128),  # 256 → 128维
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # 再次Dropout增强泛化能力
            nn.Linear(128, 10)  # 128 → 10维（对应10个类别）
        )

    def forward(self, x):
        """
        前向传播：定义数据在模型中的流动路径
        :param x: 输入张量（batch_size × 1 × 28 × 28）
        :return: 输出张量（batch_size × 10）
        """
        x = self.conv_block1(x)  # 卷积块1处理
        x = self.conv_block2(x)  # 卷积块2处理
        x = self.conv_block3(x)  # 卷积块3处理
        x = x.view(-1, 128 * 3 * 3)  # 展平：(batch_size × 128 × 3 × 3) → (batch_size × 1152)
        x = self.fc_layers(x)  # 全连接层处理
        return x  # 返回10个类别的预测值（未归一化）


class ModelTrainer:

    def __init__(self, model, trainloader, testloader):
        """
        初始化训练器
        :param model: 待训练的CNN模型
        :param trainloader: 训练集加载器
        :param testloader: 测试集加载器
        """
        self.model = model.to(DEVICE)  # 将模型移到指定设备（GPU/CPU）
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()  # 损失函数：交叉熵（分类任务专用）
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器：Adam（自适应学习率）
        # 学习率调度器：余弦退火（T_max=10→10轮后学习率降为初始值的一半）
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10)
        self.best_accuracy = 0.0  # 记录最佳测试集准确率（用于保存最优模型）
        self.final_test_acc = 0.0  # 保存最终测试集准确率（用于混淆矩阵标题）

    def train_epoch(self):
        """
        单轮训练：遍历一次训练集，更新模型参数
        :return: avg_loss - 本轮平均损失，accuracy - 本轮训练集准确率
        """
        self.model.train()  # 切换到训练模式（启用Dropout、BatchNorm训练模式）
        running_loss = 0.0  # 累计损失
        correct = 0  # 正确预测数
        total = 0  # 总样本数

        # 遍历训练集批次
        for inputs, targets in self.trainloader:
            # 将输入和标签移到指定设备
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            self.optimizer.zero_grad()  # 梯度清零（避免累积上一轮梯度）
            outputs = self.model(inputs)  # 前向传播：输入→模型→输出预测值
            loss = self.criterion(outputs, targets)  # 计算损失（预测值vs真实标签）
            loss.backward()  # 反向传播：计算梯度
            self.optimizer.step()  # 优化器更新参数

            # 累计损失（loss.item()获取张量的数值）
            running_loss += loss.item()
            # 计算准确率：outputs.max(1)取每个样本预测概率最大的类别
            _, predicted = outputs.max(1)
            total += targets.size(0)  # 累计总样本数
            correct += predicted.eq(targets).sum().item()  # 累计正确数

        # 计算本轮平均损失（总损失/批次数量）
        avg_loss = running_loss / len(self.trainloader)
        # 计算本轮准确率（正确数/总数 × 100→百分比）
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def evaluate(self):
        """
        测试集评估：不更新参数，仅计算损失和准确率，记录预测结果
        :return: avg_loss - 测试集平均损失，accuracy - 测试集准确率，preds - 所有预测标签，targets - 所有真实标签
        """
        self.model.eval()  # 切换到评估模式（禁用Dropout、BatchNorm固定）
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []  # 保存所有预测标签
        all_targets = []  # 保存所有真实标签

        with torch.no_grad():  # 禁用梯度计算（加速，节省内存）
            # 遍历测试集批次
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 保存预测和真实标签（转CPU→numpy，便于后续计算混淆矩阵）
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 计算平均损失和准确率
        avg_loss = test_loss / len(self.testloader)
        accuracy = 100. * correct / total
        self.final_test_acc = accuracy  # 保存最终测试集准确率
        return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)

    def train(self, epochs=15, patience=5):
        """
        完整训练流程：多轮训练+早停+学习率调度+训练过程可视化
        :param epochs: 最大训练轮数，默认15
        :param patience: 早停耐心值（连续patience轮无提升则停止），默认5
        :return: 训练/测试的损失和准确率列表（用于绘图）
        """
        # 初始化列表：记录每轮的损失和准确率
        train_losses, train_accs, test_losses, test_accs = [], [], [], []
        early_stop_counter = 0  # 早停计数器（连续无提升的轮数）

        print("\n========== 开始训练 ==========")
        # 遍历每一轮训练
        for epoch in range(epochs):
            # 单轮训练：返回训练损失和准确率
            train_loss, train_acc = self.train_epoch()
            # 单轮评估：返回测试损失和准确率
            test_loss, test_acc, _, _ = self.evaluate()

            # 记录本轮结果
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            # 打印本轮训练/测试结果（格式化输出，对齐更美观）
            print(f'Epoch {epoch + 1:2d}/{epochs} | 训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%')
            print(f'{"":12} | 测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.2f}%\n')

            # 学习率调度：每轮更新学习率
            self.scheduler.step()

            # 早停机制：判断是否更新最佳模型
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc  # 更新最佳准确率
                self.save_model('best_model.pth')  # 保存最优模型
                early_stop_counter = 0  # 重置早停计数器
            else:
                early_stop_counter += 1  # 计数器+1
                # 若连续patience轮无提升，触发早停
                if early_stop_counter >= patience:
                    print(f"早停触发！在第 {epoch + 1} 轮停止训练（最佳准确率: {self.best_accuracy:.2f}%）")
                    break  # 退出训练循环

        # 绘制训练过程曲线（损失+准确率）
        self.plot_training_curves(train_losses, train_accs, test_losses, test_accs)
        return train_losses, train_accs, test_losses, test_accs

    def plot_training_curves(self, train_losses, train_accs, test_losses, test_accs):
        """
        绘制训练过程的损失和准确率曲线图（保存图片）
        作用：直观查看模型训练趋势（是否过拟合、收敛情况）
        :param train_losses: 训练损失列表
        :param train_accs: 训练准确率列表
        :param test_losses: 测试损失列表
        :param test_accs: 测试准确率列表
        """
        # 创建1行2列的子图（损失曲线+准确率曲线）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('模型训练过程', fontsize=16)

        # 子图1：损失曲线
        ax1.plot(train_losses, 'b-', label='训练损失', linewidth=2)  # 蓝色线：训练损失
        ax1.plot(test_losses, 'r-', label='测试损失', linewidth=2)  # 红色线：测试损失
        ax1.set_xlabel('训练轮数', fontsize=12)
        ax1.set_ylabel('损失值', fontsize=12)
        ax1.set_title('训练/测试损失变化', fontsize=14)
        ax1.legend(fontsize=10)  # 图例
        ax1.grid(True, alpha=0.3)  # 网格（透明度0.3，更美观）

        # 子图2：准确率曲线
        ax2.plot(train_accs, 'b-', label='训练准确率', linewidth=2)  # 蓝色线：训练准确率
        ax2.plot(test_accs, 'r-', label='测试准确率', linewidth=2)  # 红色线：测试准确率
        ax2.set_xlabel('训练轮数', fontsize=12)
        ax2.set_ylabel('准确率(%)', fontsize=12)
        ax2.set_title('训练/测试准确率变化', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        # 保存图片
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ 已保存训练损失/准确率曲线图: training_curves.png")

    def save_model(self, path):
        """
        保存模型权重（含模型参数、优化器参数、最佳准确率）
        :param path: 保存路径（如'best_model.pth'）
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),  # 模型参数
            'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器参数
            'best_accuracy': self.best_accuracy  # 最佳准确率（便于后续查看）
        }, path)
        print(f"✅ 最佳模型已保存: {path} (准确率: {self.best_accuracy:.2f}%)")

    def analyze_confusion_matrix(self, classes):
        """
        计算并分析混淆矩阵：
        1. 绘制混淆矩阵热力图
        2. 计算每个类别的准确率、精确率
        3. 分析类别混淆情况
        4. 输出完整分类报告
        :param classes: 类别名称列表（中文）
        :return: cm - 混淆矩阵，class_report - 分类报告字典
        """
        # 获取测试集的预测标签和真实标签
        _, _, preds, targets = self.evaluate()

        # 1. 计算混淆矩阵：行=真实类别，列=预测类别
        cm = confusion_matrix(targets, preds)

        # 2. 绘制混淆矩阵热力图
        plt.figure(figsize=(12, 10))
        # sns.heatmap：绘制热力图，annot=True=显示数值，fmt='d'=整数格式，cmap='Blues'=蓝色系
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    annot_kws={"size": 10})  # 数值字体大小
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        # 标题：包含测试集整体准确率
        plt.title(f'Fashion-MNIST 混淆矩阵 (测试集准确率: {self.final_test_acc:.2f}%)', fontsize=14)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 3. 详细分析混淆情况
        print("\n========== 混淆矩阵分析 ==========")
        print(f"📊 模型在测试集上的整体准确率: {self.final_test_acc:.2f}%")
        print("\n🔍 类别混淆详情:")
        print("-" * 60)

        # 生成分类报告（包含精确率、召回率、F1）
        class_report = classification_report(targets, preds, target_names=classes, output_dict=True)
        # 遍历每个类别，分析详细指标
        for i, cls in enumerate(classes):
            correct = cm[i, i]  # 真实为i且预测为i的数量（正确数）
            total_true = cm[i, :].sum()  # 真实为i的总数
            total_pred = cm[:, i].sum()  # 预测为i的总数

            # 类别准确率（召回率）：正确数/真实总数
            acc = 100. * correct / total_true if total_true > 0 else 0
            # 精确率：正确数/预测总数
            precision = 100. * correct / total_pred if total_pred > 0 else 0

            # 打印当前类别的指标
            print(f"类别 {i:2d} [{cls}]:")
            print(f"  - 准确率(召回率): {acc:.2f}% ({correct}/{total_true})")
            print(f"  - 精确率: {precision:.2f}% ({correct}/{total_pred})")

            # 找出当前类别最易混淆的类别
            cm_copy = cm[i].copy()
            cm_copy[i] = 0  # 排除自身（只看混淆的类别）
            max_confuse_idx = np.argmax(cm_copy)  # 混淆数最多的类别索引
            max_confuse_count = cm_copy[max_confuse_idx]  # 混淆数量
            if max_confuse_count > 0:
                print(f"  - 最易混淆为: {classes[max_confuse_idx]} (数量: {max_confuse_count})")
            print()  # 空行分隔

        # 输出完整分类报告（文本格式）
        print("📋 完整分类报告:")
        print("-" * 60)
        print(classification_report(targets, preds, target_names=classes, digits=2))

        return cm, class_report


class ImagePredictor:

    def __init__(self, model, classes):
        """
        初始化预测器
        :param model: 训练好的CNN模型
        :param classes: 类别名称列表
        """
        self.model = model.to(DEVICE)  # 模型移到指定设备
        self.model.eval()  # 切换到评估模式
        self.classes = classes  # 类别名称

    def preprocess_image(self, image_path):
        """
        预处理自定义图片（匹配模型输入格式）
        :param image_path: 图片路径
        :return: 预处理后的张量（batch_size × 1 × 28 × 28），失败返回None
        """
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"❌ 图片文件不存在: {image_path}")
            return None

        try:
            # 1. 打开图片并转换为灰度图（匹配Fashion-MNIST格式）
            image = Image.open(image_path).convert('L')
            # 2. 反转灰度：Fashion-MNIST是白底黑图，外部图片通常是黑底白图，反转后匹配
            image = ImageOps.invert(image)

            # 3. 预处理流水线（与训练集一致）
            preprocess = transforms.Compose([
                transforms.Resize((32, 32)),  # 先放大到32×32（避免直接缩28×28变形）
                transforms.CenterCrop((28, 28)),  # 中心裁剪到28×28
                transforms.ToTensor(),  # 转张量
                transforms.Normalize((0.5,), (0.5,))  # 标准化（与训练集一致）
            ])

            # 4. 应用预处理并添加batch维度（模型要求输入是批量，即使单张图片）
            tensor_image = preprocess(image).unsqueeze(0)
            return tensor_image

        except Exception as e:
            # 捕获所有异常（如图片格式错误、路径错误等）
            print(f"❌ 图片处理失败: {e}")
            return None

    def predict_image(self, image_tensor):
        """
        预测自定义图片
        :param image_tensor: 预处理后的张量
        :return: predicted_class - 预测类别名称，confidence_score - 置信度（百分比）
        """
        if image_tensor is None:
            return None, 0.0

        with torch.no_grad():  # 禁用梯度计算
            # 前向传播：输入→模型→输出预测值
            outputs = self.model(image_tensor.to(DEVICE))
            # 转换为概率（softmax归一化，使所有类别概率和为1）
            probs = F.softmax(outputs, dim=1)
            # 获取最大概率的置信度和类别索引
            confidence, predicted = torch.max(probs, 1)

        # 转换为类别名称和置信度百分比
        predicted_class = self.classes[predicted.item()]
        confidence_score = confidence.item() * 100

        # 打印预测结果
        print("\n========== 自定义图片预测结果 ==========")
        print(f"预测类别: {predicted_class}")
        print(f"置信度: {confidence_score:.2f}%")
        print("\n所有类别置信度（降序）:")

        # 将概率转换为numpy数组，便于排序
        prob_list = probs.squeeze().cpu().numpy()
        # 按概率从高到低排序，获取索引
        sorted_indices = np.argsort(prob_list)[::-1]
        # 遍历排序后的索引，打印每个类别的置信度
        for idx in sorted_indices:
            print(f"  {self.classes[idx]:<8}: {prob_list[idx] * 100:.2f}%")

        return predicted_class, confidence_score


if __name__ == "__main__":
    """
    主程序流程：
    1. 数据准备 → 2. 模型定义与结构可视化 → 3. 模型训练 → 4. 混淆矩阵分析 → 5. 自定义图片预测
    """
    # 1. 数据准备与随机样本可视化
    data_processor = DataProcessor(batch_size=64)  # 创建数据处理器（批量64）
    trainloader, testloader = data_processor.load_data()  # 加载数据
    data_processor.visualize_random_samples()  # 可视化每个类别样本

    # 2. 模型定义与结构描述
    model = FashionCNN()  # 创建CNN模型
    print("\n========== 模型结构描述 ==========")
    # 文字描述模型结构（便于理解）
    print("📌 FashionCNN 模型结构:")
    print("  输入: 1×28×28 灰度图像")
    print("  卷积块1: Conv2d(32) → BatchNorm → ReLU → MaxPool → 输出32×14×14")
    print("  卷积块2: Conv2d(64) → BatchNorm → ReLU → MaxPool → 输出64×7×7")
    print("  卷积块3: Conv2d(128) → BatchNorm → ReLU → MaxPool → 输出128×3×3")
    print("  全连接层: 128×3×3 → 256 → Dropout(0.3) → 128 → Dropout(0.3) → 10（输出）")

    # 图表形式：使用torchsummary输出层维度和参数（直观展示模型结构）
    print("\n📌 模型层维度详情:")
    summary(model, input_size=(1, 28, 28))  # 输入尺寸：1×28×28（灰度图）

    # 3. 模型训练
    trainer = ModelTrainer(model, trainloader, testloader)  # 创建训练器
    trainer.train(epochs=15, patience=5)  # 开始训练（15轮，早停耐心值5）

    # 4. 混淆矩阵分析 + 测试集准确率输出
    trainer.analyze_confusion_matrix(data_processor.classes)

    # 5. 自定义图片预测
    predictor = ImagePredictor(model, data_processor.classes)  # 创建预测器
    image_path = "111.jpg"  # 自定义图片路径（需放在脚本同目录）
    if os.path.exists(image_path):
        processed_img = predictor.preprocess_image(image_path)  # 预处理图片
        if processed_img is not None:
            predictor.predict_image(processed_img)  # 预测图片
    else:
        print(f"\n⚠️  自定义图片 {image_path} 不存在，跳过预测")

    # 打印完成提示（列出生成的文件）
    print("\n🎉 所有分析任务完成！生成的文件:")
    print("  - random_class_samples.png (每个类别随机样本)")
    print("  - training_curves.png (损失/准确率曲线)")
    print("  - confusion_matrix.png (混淆矩阵)")
    print("  - best_model.pth (最佳模型权重)")