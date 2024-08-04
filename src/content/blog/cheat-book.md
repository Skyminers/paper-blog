---
author: Sky_miner
pubDatetime: 2024-08-04T01:48:39.000+08:00
# modDatetime:
title: 面试 Cheat Book
featured: false
draft: false
tags:
  - docs
description: 网上总结的面试 Cheat Book
---

## Table of contents

## C++ 相关

### C++ 虚函数

一般定义虚函数时是为了多态，允许基类的虚函数指针调用不同子类对该虚函数实现的不同功能。虚函数分为了虚函数和纯虚函数，虚函数是有默认的功能实现的，纯虚函数是没有实现的，要求子类必须实现这个函数。

纯虚函数的定义方式是在定义的语句后面加`=0`，例如：`virtual void funtion1()=0`

## Python 相关

### Python 可变对象与不可变对象

不可变对象是不可变的，当修改这类对象时实际上是创建了一个新的对象并修改引用，并且对原来的不可变对象进行回收。可变对象是可变的，修改的话就是直接修改这个对象。

- 不可变对象：int, float, str, tuple
- 可变对象：list, dict, set

## 机器学习相关

### Batchnorm 与 LayerNorm

作用都是将输入的数据归一化，均值变为 $0$，方差变为 $1$。

Batchnorm 是对每个 batch 的某一个维度组合成的特征进行归一化，LayerNorm 是对每个样本的特征进行归一化。Transformers 中使用 LayerNorm，因为理解上来说不同 Batch 样本中的某个特征不存在相关性，而同一个样本中的特征之间存在相关性。大部分样本 embedding 是没有意义的，少部分是有意义的，如果使用 Batchnorm 可能反而会破坏这些有意义的 embedding。

![](@assets/images/cheat-book/batch-layer-norm.png)

### Batchnorm 有哪些参数

可学习参数：

- $\gamma$：缩放因子，控制标准化后的数据 scale，训练过程中每一个特征通道都有一个可学习的缩放因子。
- $\beta$：偏移因子，控制标准化后的数据 shift，训练过程中每一个特征通道都有一个可学习的偏移因子。

不可学习参数：全局均值和全局方差。

### Vit 为什么加入 cls 分类 token？相比直接在最后用第一个有什么好处？

Token 可以捕获全局的信息，如果直接用第一个的话可能会跟第一个 Patch 有很强的关系，无法有效捕获其他 Patch 的信息。

### Transformer 为什么要除以根号 dk

因为在计算 Attention 时，计算的是 $Q \cdot K^T$，如果不除以根号 dk，那么 $Q \cdot K^T$ 的值会很大，导致 Softmax 后的值很小，这样会导致梯度消失。

## 智力/算法题

### 如何在 1T 的数据中寻找中位数

快速选择：每次找中间点，然后遍历一遍，将小于中间点的放左边，大于中间点的放右边，然后递归。
