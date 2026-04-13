import mysql.connector
from hashlib import sha256

# 数据库配置，建议根据实际进行修改
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '54088wsj',  # 请替换为你的数据库密码
    'database': 'road_damage_db'
}

class DBManager:
    @staticmethod
    def get_connection():
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            return conn
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None

    @staticmethod
    def hash_password(password):
        """对密码进行SHA-256加密"""
        return sha256(password.encode('utf-8')).hexdigest()

    @classmethod
    def register_user(cls, username, password):
        """用户注册"""
        conn = cls.get_connection()
        if not conn:
            return False, "无法连接到数据库"
        
        try:
            cursor = conn.cursor()
            hashed_pwd = cls.hash_password(password)
            sql = "INSERT INTO users (username, password) VALUES (%s, %s)"
            cursor.execute(sql, (username, hashed_pwd))
            conn.commit()
            return True, "注册成功"
        except mysql.connector.IntegrityError:
            return False, "用户名已存在"
        except Exception as e:
            return False, f"注册失败: {str(e)}"
        finally:
            conn.close()

    @classmethod
    def login_user(cls, username, password):
        """用户登录"""
        conn = cls.get_connection()
        if not conn:
            return False, "无法连接到数据库"
        
        try:
            cursor = conn.cursor()
            hashed_pwd = cls.hash_password(password)
            sql = "SELECT * FROM users WHERE username = %s AND password = %s"
            cursor.execute(sql, (username, hashed_pwd))
            user = cursor.fetchone()
            if user:
                return True, "登录成功"
            else:
                return False, "用户名或密码错误"
        except Exception as e:
            return False, f"登录异常: {str(e)}"
        finally:
            conn.close()

    @classmethod
    def add_segmentation_record(cls, username, task_type, original_path, result_path):
        """新增分割记录"""
        conn = cls.get_connection()
        if not conn:
            return False, "无法连接到数据库"
        
        try:
            cursor = conn.cursor()
            sql = """
                INSERT INTO segmentation_records (username, task_type, original_path, result_path) 
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (username, task_type, original_path, result_path))
            conn.commit()
            return True, "记录保存成功"
        except Exception as e:
            return False, f"记录保存失败: {str(e)}"
        finally:
            conn.close()

    @classmethod
    def get_user_records(cls, username):
        """获取指定用户的分割记录"""
        conn = cls.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(dictionary=True)
            sql = "SELECT id, task_type, original_path, result_path, created_at FROM segmentation_records WHERE username = %s ORDER BY created_at DESC"
            cursor.execute(sql, (username,))
            return cursor.fetchall()
        except Exception as e:
            print(f"获取记录失败: {e}")
            return []
        finally:
            conn.close()
