# log

#### 7/15  

pg  
a.我開了新的gcp vm給rl用，快很多  
b.認真trace karpathy policy gradient gist，發現他在discount rewards 做的處理很特殊，有做clipping   
c.用clipping 後reward 仍然很小，剛剛再看一遍投影片，認為preprocessing我還少做residual ，明天加上看效果如何    

dqn  
a.把教授投影片看好多次，大致上只要看algorithm 那裡就好  
b.今天花很多很多時間處理wrapper 的bug，後來直接拿deepmind原本的wrapper覆蓋(因為他們本來就是用deepmind 的wrapper，然後加幾個function，會有bug我爬一下github好像是版本的問題，有prefix等等)


projects  
a.今天想到有個很容易做的projects，path planning(grid version)，然後用走迷宮方式規劃，並且加一點點方形障礙物  
b.pybullet projects  
https://kknews.cc/zh-tw/other/vkm4pg2.html

