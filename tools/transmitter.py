import socket
import json
import torch
import numpy as np
from mlsocket import MLSocket
import time
import threading
class Transmitter():
    """
    Trasmitter using  mlsocket.
    """
    def __init__(self, reciever_ip:str, reciever_port:int):
        """
        Args:
            reciever_ip: IP of the reciever.
            reciever_port: Port of the reciever.
        """
        super().__init__()
        self.reciever_ip = reciever_ip
        self.reciever_port = reciever_port
        self.pcd = None
        self.s = MLSocket()
        self.pred_dict = None
        self.started = False
        self.thread = None

        #try:
        #    self.sock.connect(self.reciever_ip)
        #except:
        #    print("Could not connect to the reciever.")

    def send_dict(self, pcd,dict,classes_to_send=None):
        """
        Send a dictionary.
        Args:
            dict: Dictionary to send.
        """
        dict_human = {"pred_boxes": [], "pred_labels": [], "pred_scores": []}
        for key, value in (dict.items()):
            if isinstance(value,torch.Tensor):
                dict[key] = value.detach().cpu().tolist()
            if isinstance(value,np.ndarray):
                dict[key] = value.tolist()
        if classes_to_send is not None:
            for i,v in enumerate(dict["pred_labels"]):
                print(v)
                if v in classes_to_send:
                    dict_human['pred_boxes'].append(dict["pred_boxes"][i])
                    dict_human['pred_labels'].append(dict["pred_labels"][i])
                    dict_human['pred_scores'].append(dict["pred_scores"][i])
            dict = dict_human
        dict["pcd"] = pcd.tolist()
        user_encode_data = json.dumps(dict).encode('utf-8')
        self.send(user_encode_data, (self.reciever_ip, self.reciever_port))
    def _check_connection(self):
        """
        Check if the connection is alive.
        """
        try:
            self.sock.sendto(b'ping', (self.reciever_ip, self.reciever_port))
            data, addr = self.sock.recvfrom(1024)
            if data.decode('utf-8') == 'pong':
                return True
            else:
                return False
        except:
            return False
    def check_connection(self):
        """
        Check if the connection is alive.
        """
        try:
            self.connect((self.reciever_ip, self.reciever_port))
        except:
            return False
    def start_transmit(self):
        """
        Start the transmission.
        """
        try: 
            self.s.connect((self.reciever_ip,self.reciever_port))
            self.thread = threading.Thread(target=self.transmit)
            self.thread.daemon = True
            self.thread.start()
            self.started = True
        except:
            print("Could not connect to the reciever.")
    def stop_transmit(self):
        """
        Stop the transmission.
        """
        if self.started:
            self.started = False
            self.thread.join()
    def transmit(self):
        """
        Transmit the data.
        """
        print("Started Transmitter to {}:{}".format(self.reciever_ip, self.reciever_port))
        while self.started:
            if self.pcd is not None and self.pred_dict is not None:
                print("SENDING DATA")
                self.s.send(self.pcd)
                try:
                    pred_arr = np.concatenate([np.array(self.pred_dict["pred_boxes"]),np.array(self.pred_dict["pred_labels"]),np.array(self.pred_dict["pred_scores"])],axis=1)
                    self.s.send(pred_arr)
                    print(f"SENT : {pred_arr.shape}")
                except:
                    pass
                self.pcd = None
                self.pred_dict = None
if __name__ == "__main__":
    #transmitter = Transmitter(reciever_ip="192.168.200.103", reciever_port= 7003)
    #
    #test_dict = {"pred_boxes":np.random.random((1,9)),"pred_scores": np.random.random((1,1)),"pred_labels": np.random.random([9])}
    #transmitter.send_dict(np.random.random((500,3)),test_dict)
    HOST = '192.168.200.103'
    PORT = 1234
    detections = 10
    pcd = np.random.random((100000,3))
    bboxes = np.random.random((detections,3))
    lbls = np.random.randint(0,10,(detections,1))
    score = np.random.randint(0,10,(detections,1))

    d = {"pred_boxes": np.random.random((1,9)).tolist(),"pred_scores": np.random.random((1,1)).tolist(),"pred_labels": np.random.random([9]).tolist()}

    with MLSocket() as s:
        s.connect((HOST, PORT)) # Connect to the port and host
        start = time.time()
        #user_encode_data = json.dumps(d).encode('utf-8')
        s.send(pcd)
        a =  np.concatenate((bboxes,lbls,score),axis=1)
        s.send(a)
        print(f"Elapsed time: {time.time() - start}")



