clc; clear all

hs = [0.08 0.92]; % hidden states

st = [0.76 0.24; 
      0.1 0.9]; % state transition

dm = [0.74 0.26; 
      0.96 0.04]; % decision matrix
dm = transpose(dm);
%% Observations:
o1 = [1 0];
o2 = [0 1];
o3 = [1 0];
%% 1st hidden state
    if o1(1,1) ==1
        dm1 = dm(1,:);
    else
        dm1 = dm(2,:);
    end
    
    X1 = [];
    for i = 1:2
    x1_1 = hs(1)*st(1,i)*dm1(i);
    x1_2 = hs(2)*st(2,i)*dm1(i);
    X1(i)= max([x1_1 x1_2]);
    end

x1 = round((X1(1,1)/sum(X1)),2);
x2 = round((X1(2)/sum(X1)),2);
first_hidden_state = [x1 x2]
%% 2nd hidden state
    if o2(1,1) ==1
        dm1 = dm(1,1:2);
    else
        dm1 = dm(2,1:2);
    end
    
    X1 = [];
    for i = 1:2
    x1_1 = x1*st(1,i)*dm1(i);
    x1_2 = x2*st(2,i)*dm1(i);
    X1(i)= max([x1_1 x1_2]);
    end

y1 = round((X1(1,1)/sum(X1)),2);
y2 = round((X1(2)/sum(X1)),2);
second_hidden_state = [y1 y2]

%% 3rd hidden state
    if o3(1,1) ==1
        dm1 = dm(1,1:2);
    else
        dm1 = dm(2,1:2);
    end
    
    X1 = [];
    for i = 1:2
    x1_1 = y1*st(1,i)*dm1(i);
    x1_2 = y2*st(2,i)*dm1(i);
    X1(i)= max([x1_1 x1_2]);
    end

z1 = round((X1(1,1)/sum(X1)),2);
z2 = round((X1(2)/sum(X1)),2);
third_hidden_state = [z1 z2]
Our_answer = z1