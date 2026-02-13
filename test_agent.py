import subprocess

# 测试不同调用方式
cmd = ["openclaw", "agent", "--message", "你是谁？回答：我是Research Agent", "--session-id", "test-basic"]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
print("Return code:", result.returncode)
print("\n=== STDOUT (最后1000字符) ===")
print(result.stdout[-1000:])
print("\n=== STDERR ===")
print(result.stderr[:500])
