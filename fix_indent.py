with open('model_visualization.py', 'r') as f:
    lines = f.readlines()

# Remove the misplaced best_precision and best_recall
new_lines = []
i = 0
while i < len(lines):
    if 'best_precision = max(results.items()' in lines[i] and i > 170 and i < 190:
        # skip this and next line
        i += 2
    elif 'print(f"Best Precision:' in lines[i] and i > 170 and i < 190:
        # skip this and next line
        i += 2
    else:
        new_lines.append(lines[i])
        i += 1

with open('model_visualization.py', 'w') as f:
    f.writelines(new_lines)