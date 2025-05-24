import cv2
import numpy as np
from collections import deque

class DSJ:                       # 併查集
    def __init__(self, size):    # 初始化，parent為從0到size - 1的list (parent[x]代表節點x的root)
        self.parent = list(range(0,size)) # 每個點的初始root是自己
    
    def find(self, x):           # 尋找x的root
        if self.parent[x] == x:  # 若自身為root則return
            return x
        self.parent[x] = self.find(self.parent[x]) # 遞迴尋找root並設定
        return self.parent[x]    # 傳回 root

    def union(self, x, y):       # 將x及y設為相同的root
        Fx = self.find(x)        # 找到x的root
        Fy = self.find(y)        # 找到y的root
        if Fx != Fy:             # 若兩者的root不相同
            self.parent[Fy] = Fx # 將Fy的root設為Fx
            return True
        else:
            return False

def get_weight(node1, node2, direction): # 取的兩節點權重
    if direction == "R&L":       # node1在右、node2在左
        edge1 = node1[:, 0, :]   # node1左邊緣的所有pixel
        edge2 = node2[:, -1, :]  # node2右邊緣的所有pixel
    elif direction == "L&R":
        edge1 = node1[:, -1, :]   
        edge2 = node2[:, 0, :]  
    elif direction == "U&D":     # node1在上、node2在下
        edge1 = node1[-1, :, :]  # node1下邊緣的所有pixel
        edge2 = node2[0, :, :]   # node2上邊緣的所有pixel
    elif direction == "D&U": 
        edge1 = node1[0, :, :]  
        edge2 = node2[-1, :, :] 
    else:
        print(456)

    edge1 = edge1.astype(np.float32) # uint8轉為float32(防止underflow)
    edge2 = edge2.astype(np.float32)
    weight = np.linalg.norm(edge1 - edge2) # 計算邊緣pixel的歐式距離
    return weight

def find_adjacent( puzzle ): # 取得依權重排序過後的所有邊
    all_edge = []
    direction = ["R&L","L&R","U&D","D&U"] 
    for i in range(len(puzzle)):
        for j in range(len(puzzle)):
            if i != j:
                all_edge.append( (i, j, direction[0], get_weight(puzzle[i],puzzle[j],direction[0])) )
                all_edge.append( (i, j, direction[1], get_weight(puzzle[i],puzzle[j],direction[1])) )
                all_edge.append( (i, j, direction[2], get_weight(puzzle[i],puzzle[j],direction[2])) )
                all_edge.append( (i, j, direction[3], get_weight(puzzle[i],puzzle[j],direction[3])) )

    all_edge.sort(key=lambda x: x[3]) # 將edge按照weight由小排到大
    return all_edge

def MST( all_edge ):  # 最小生成樹(使用Kruskal's algorithm)
    direction = ["R&L","U&D"] 
    direction_reverse = ["L&R","D&U"]
    Mst = {i:[] for i in range(9*16)} # 建立連接圖
    dsj = DSJ(9*16)  # 建立併查集
    for edge in all_edge: # edge = (node1 ,node2, direction, weight)
        if dsj.union(edge[0], edge[1]):            # True表示兩node不會形成circle(將其相連)
            Mst[edge[0]].append((edge[1],edge[2])) # 記錄連接的node以及其相對方向
            '''
            if (edge[2] == direction[0]):          # 建立雙向邊 (無向圖)
                Mst[edge[1]].append((edge[0],direction_reverse[0]))
            elif (edge[2] == direction[1]):
                Mst[edge[1]].append((edge[0],direction_reverse[1]))
            else:
                print(1)
            '''

    print(Mst)
    return Mst

def traverse_MST( Mst, puzzle, pixel_size ):
    move = {"R&L":(-1,0), "L&R":(1,0), "U&D":(0,1), "D&U":(0,-1)}
    puzzle_position = {} 
    visited = set()
    queue = deque()
    
    # 從拼圖 0 開始，設為 (0, 0) 
    queue.append((0, 0, 0))  # (index, x, y)
    puzzle_position[0] = (0,0)
    visited.add(0)
    while queue: # BFS
        current, x, y = queue.popleft() 
        for neighbor, direction in Mst[current]:  # Mst = {edge:[(連接的edge,方向)]}
            if neighbor in visited:
                continue
            visited.add(neighbor)
            #print(move[direction])
            new_x = x + move[direction][0]
            new_y = y + move[direction][1]
            #if (new_x, new_y) == (1,-3):
            #    print( current , neighbor, direction, new_x, new_y )
            puzzle_position[neighbor] = (new_x, new_y)
            queue.append((neighbor, new_x, new_y))

    # 移動所有座標使左上角為 (0,0)
    min_x = min(x for x, y in puzzle_position.values())  # 取得最左上角的(x,y)座標
    min_y = min(y for x, y in puzzle_position.values())
    for index in range(len(puzzle_position)):
        puzzle_position[index] = ( puzzle_position[index][0] - min_x, puzzle_position[index][1] - min_y)
    
    output_img = np.zeros((1080, 1920, 3), dtype=np.uint8)  # (高,寬,channel)建立一張「空的彩色圖片（黑圖）」
    
    # 根據位置放入拼圖
    for i in range(9*16):
        x, y = puzzle_position[i]
        output_img[y*pixel_size:(y+1)*pixel_size, x*pixel_size:(x+1)*pixel_size] = puzzle[i]
    
    # 輸出結果圖
    cv2.imwrite("One_Piece1_result.bmp", output_img)

if __name__ == '__main__':
    puzzle = []         # 拿來裝完整讀入拼圖的List
    pixel_size = 120    # 每塊拼圖的pixel大小
    img = cv2.imread("One_Piece1.bmp", cv2.IMREAD_COLOR) # 讀入bmp檔
    for y in range(9):         # 9行  (高)
        for x in range(16):    # 16列 (寬)
            a_piece_of_puzzle = img[y*pixel_size:(y+1)*pixel_size, 
                                    x*pixel_size:(x+1)*pixel_size] # 取得一塊塊的拼圖(每個拼圖大小120*120)
            puzzle.append(a_piece_of_puzzle) # 將每塊拼圖加入List中(共16*9=144塊) 

    all_edge = find_adjacent(puzzle) 
    print(len(all_edge))
    #Mst = MST(all_edge)
    #traverse_MST( Mst, puzzle, pixel_size )



    
    
