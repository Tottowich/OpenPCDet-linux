
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class TimeLogger():
    def __init__(self):
        super().__init__()
        self.time_dict = {}
        self.time_pd = None

    def create_metric(self, name: str):
        self.time_dict[name] = []
    
    def log_time(self, name: str, time: float):
        self.time_dict[name].append(time)
    
    def visualize_results(self):
        time_averages = {}
        for key in self.time_dict:
            
            plt.plot(range(0,len(self.time_dict [key])),self.time_dict[key])
            time_averages[key] = np.mean(self.time_dict[key])
            #plt.show()
        time_pd = pd.DataFrame(time_averages,index=[0])
        print(f"time_pd: {time_pd}")

if __name__ == "__main__":
    T = TimeLogger()
    data = {'a': np.random.rand(10), 'b': np.random.rand(10), 'c': np.random.rand(10)}
    T.create_metric('a')
    T.create_metric('b')
    T.create_metric('c')
    T.log_time('a', data['a'])
    T.log_time('b', data['b'])
    T.log_time('c', data['c'])
    T.visualize_results()