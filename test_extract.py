import subprocess
import json

cmd = ["openclaw", "agent", "--message", "回答JSON: {\"action\": \"测试\", \"data\": [1,2,3]}", "--session-id", "test-json"]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
stdout = result.stdout

print("="*60)
print("找所有JSON块")
print("="*60)

import re
pattern = r'\{[^{}]*\}'
matches = re.findall(pattern, stdout)

print(f"找到 {len(matches)} 个JSON块\n")

# 从后往前试
for i, match in enumerate(reversed(matches)):
    try:
        data = json.loads(match)
        print(f"#{len(matches)-i}: {data}")
        break  # 第一个成功的JSON就是最后一个块
    except json.JSONDecodeError as e:
        print(f"#{len(matches)-i}: 无效JSON")
