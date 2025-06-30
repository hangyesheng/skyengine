
# 测试转化后的数据格式能否正常启动系统

from dataset.convert.convert import generate_job_config,generate_map_config
from sky_simulator.lifecycle.bootstrap import bootstrap

if __name__ == '__main__':
    generate_job_config()
    bootstrap()
