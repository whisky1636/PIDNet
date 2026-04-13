-- 道路病害分割系统 数据库初始化脚本
-- 创建数据库
CREATE DATABASE IF NOT EXISTS road_damage_db DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE road_damage_db;

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE COMMENT '用户名',
    password VARCHAR(255) NOT NULL COMMENT '加密后的密码',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 创建分割记录表
CREATE TABLE IF NOT EXISTS segmentation_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL COMMENT '执行分割的用户',
    task_type VARCHAR(20) NOT NULL COMMENT '任务类型: image, folder, video',
    original_path TEXT NOT NULL COMMENT '原始文件或目录路径',
    result_path TEXT NOT NULL COMMENT '分割结果保存的文件或目录路径',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
