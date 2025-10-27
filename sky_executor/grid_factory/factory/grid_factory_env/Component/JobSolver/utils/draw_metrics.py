'''
@Project ：SkyEngine 
@File    ：draw_metrics.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/27 22:04
'''
import matplotlib.pyplot as plt

def plot_machine_gantt(machine_schedule, machines, title="Machine Gantt"):
    fig, ax = plt.subplots(figsize=(10, 3 + len(machines) * 0.25))
    y_labels = []
    for i, m in enumerate(machines):
        tasks = sorted(machine_schedule.get(m.m_id, []), key=lambda x: x[0])
        for (s, e, jid, oid) in tasks:
            ax.barh(i, e - s, left=s, height=0.6)
            ax.text(s + 0.02 * (e - s + 1), i, f"J{jid}-O{oid}", va='center', fontsize=8)
        y_labels.append(f"M{m.m_id}")
    ax.set_yticks(list(range(len(machines))))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_machine_util_bar(stats):
    util = stats['machine_util']
    mids = sorted(util.keys())
    vals = [util[m] for m in mids]
    plt.figure(figsize=(6, 3))
    plt.bar([f"M{m}" for m in mids], vals)
    plt.title("Machine Utilization")
    plt.ylim(0, 1)
    plt.show()

