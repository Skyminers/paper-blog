---
author: Sky_miner
pubDatetime: 2024-08-01T20:41:09+08:00
# modDatetime:
title: LeetCode 热题 100 一句话题解集(更新中)
featured: true
draft: false
tags:
  - leetcode
  - algorithm
description: LeetCode 热题 100 一句话题解集，简要概括题目思路解法，也会提供对应代码。
---

一句话题解，简要概括题目思路解法，也会提供对应代码。

## Table of contents

## 1. [两数之和](https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked)

一个数字确定时另一个数字的值也就确定了，可以用 hashmap 判断另一个数字是否存在，时间复杂度 O(n)。C++ 中可以用 `unordered_map` 实现。

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> mp;
        for (int i  = 0; i < nums.size(); ++ i) {
            auto it = mp.find(target - nums[i]);
            if (it != mp.end()) {
                return {it->second, i};
            }
            mp[nums[i]] = i;
        }
        return {};
    }
};
```

---

## 11. [盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/?envType=study-plan-v2&envId=top-100-liked)

贪心，从两侧用双指针向中间靠拢，每次移动更低的一边，这样才更可能向更好的方向发展。

```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i = 0, j = height.size()-1;
        int ans = 0;
        while (i < j) {
            ans = max(ans, min(height[i], height[j]) * (j-i));
            if (height[i] < height[j]) ++ i;
            else --j;
        }
        return ans;
    }
};
```

---

## 15. [三数之和](https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked)

首先排序，然后枚举三元组的第一个数字，再枚举第二个数字，第三个数字会在第二个数字从前往后枚举的过程中逐渐从后往前移动，维护指针即可。

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());

        vector<vector<int>> ans;
        ans.clear();

        for (int i = 0;i < nums.size(); ++ i) {
            if (i != 0 && nums[i] == nums[i-1]) continue;

            int k = nums.size()-1;

            for (int j = i+1; j < nums.size(); ++ j) {

                if (j > i+1 && nums[j] == nums[j-1]) continue;
                while(k > j && nums[i] + nums[j] + nums[k] > 0) -- k;
                if (k <= j) break;

                if (nums[i] + nums[j] + nums[k] == 0) {
                    vector<int> tmp; tmp.clear();
                    tmp.push_back(nums[i]);
                    tmp.push_back(nums[j]);
                    tmp.push_back(nums[k]);
                    ans.push_back(tmp);
                }
            }
        }
        return ans;
    }
};
```

---

## 42. [接雨水](https://leetcode.cn/problems/trapping-rain-water/?envType=study-plan-v2&envId=top-100-liked)

可以看到，在从左向右移动的过程中，每次最高点更新时就会有一个新的接雨水的坑出现。所以循环两次，第一次从左向右，第二次从右向左，如果最高点有更新的话就是出现了新的水坑。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int maxHeightIdx = 0;
        int sum = 0, ans = 0;
        for (int i = 1; i < height.size(); ++ i) {
            if (height[i] >= height[maxHeightIdx]) {
                ans += height[maxHeightIdx] * (i - maxHeightIdx - 1) - sum;
                sum = 0;
                maxHeightIdx = i;
            } else {
                sum += height[i];
            }
        }
        int middle = maxHeightIdx;
        maxHeightIdx = height.size()-1;
        sum = 0;
        for (int i = height.size()-2; i >= middle; -- i) {
            if (height[i] >= height[maxHeightIdx]) {
                ans += height[maxHeightIdx] * (maxHeightIdx - i - 1) - sum;
                sum = 0;
                maxHeightIdx = i;
            } else {
                sum += height[i];
            }
        }
        return ans;
    }
};
```

---

## 49. [字母异位词分组](https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&envId=top-100-liked)

将所有字符串排序，此时相等的字符串就是字母异位词，用排序后的字符串作为 key，原字符串作为 value 存入 hashmap 中，最后将 hashmap 中的 value 取出即可。用 C++ 的 `unordered_map` 实现。

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        int len = strs.size();
        unordered_map<string, vector<string>> mp;
        int cnt = 0;
        for (auto it: strs) {
            string sortResult = it;
            sort(sortResult.begin(), sortResult.end());
            mp[sortResult].push_back(it);
        }
        vector<vector<string>> ret;
        for (auto it : mp) {
            ret.push_back(it.second);
        }
        return ret;
    }
};
```

---

## 51. [N 皇后](https://leetcode.cn/problems/n-queens/description/?envType=study-plan-v2&envId=top-100-liked)

暴搜即可

```cpp
class Solution {
    vector<vector<string>> ans;
    int mat[10][10];
public:
    vector<vector<string>> solveNQueens(int n) {
        ans.clear();
        memset(mat, 0, sizeof mat);
        dfs(n, 1, 1);
        return ans;
    }

    void dfs(int n, int rows, int idx) {
        if (idx == n+1) {
            return ;
        }
        if (rows > n) {
            vector<string> res;
            res.clear();
            for (int i = 1; i <= n; ++ i) {
                string row_ans = "";
                for (int j = 1; j <= n; ++ j) {
                    if (mat[i][j]) row_ans += "Q";
                    else row_ans += ".";
                }
                res.push_back(row_ans);
            }
            ans.push_back(res);
            return ;
        }
        if (check(n, rows, idx)) {
            mat[rows][idx] = 1;
            dfs(n, rows+1, 1);
            mat[rows][idx] = 0;
        }
        dfs(n, rows, idx+1);
    }

    bool check(int n, int x, int y) {
        for (int i = 1; x-i > 0 && y - i > 0; ++ i) {
            if (mat[x - i][y - i]) return false;
        }
        for (int i = 1; x-i > 0 && y + i <= n; ++ i) {
            if (mat[x - i][y + i]) return false;
        }
        for (int i = 1; i < x; ++ i) {
            if (mat[i][y]) return false;
        }
        return true;
    }
};
```

---

## 108. [将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description/?envType=study-plan-v2&envId=top-100-liked)

通过 dfs 进行递归建树，每次取中间的数字作为根节点，然后递归建立左右子树。

```cpp
class Solution {
public:
    TreeNode* dfs(int l, int r, vector<int>& nums) {
        if (l > r) return NULL;
        int mid = l+r >> 1;
        TreeNode* rt = new TreeNode();
        rt->val = nums[mid];
        rt->left = dfs(l, mid-1, nums);
        rt->right = dfs(mid+1, r, nums);
        return rt;
    }
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return dfs(0, nums.size()-1, nums);
    }
};
```

---

## 128. [最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/?envType=study-plan-v2&envId=top-100-liked)

将所有数字插入到 HashMap 内后，通过枚举的方式寻找连续段的起点，然后通过循环找到终点即可。

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> num_set;
        for (auto x: nums) {
            num_set.insert(x);
        }
        int ans = 0;
        for (auto x: num_set) {
            if (!num_set.count(x-1)) {
                int cur = x;
                int ret = 1;
                while(num_set.count(cur+1)) {
                    ++ cur;
                    ++ ret;
                }
                ans = max(ans, ret);
            }
        }
        return ans;
    }
};
```

---

## 146. [LRU 缓存](https://leetcode.cn/problems/lru-cache/?envType=study-plan-v2&envId=top-100-liked)

hash 表维护每个 key 对应的指针，双向链表维护数据的顺序（强烈建议使用头尾实节点来写）。

```cpp
struct Node {
    int key, value;
    Node *nxt, *pre;
};

class LRUCache {
    unordered_map<int, Node*> hash;
    int cap, siz;
    Node *head, *tail;
public:
    LRUCache(int capacity) {
        cap = capacity;
        siz = 0;
        hash.clear();
        head = new Node();
        tail = new Node();
        head->key = head->value = -1;
        head->pre = NULL;
        head->nxt = tail;

        tail->key = tail->value = -1;
        tail->pre = head;
        tail->nxt = NULL;
    }

    int get(int key) {
        if (hash.contains(key)) {
            Node* node = hash[key];
            update(node, node->value);
            return node->value;
        }
        return -1;
    }

    void put(int key, int value) {
        if (hash.contains(key)) {
            Node* node = hash[key];
            update(hash[key], value);
        } else {
            hash[key] = insert(key, value);
            if (++siz > cap) {
                int delete_key = delete_last();
                hash.erase(delete_key);
            }
        }
    }

    Node* insert(int key, int value) {
        Node *ret = new Node;
        ret->value = value;
        ret->key = key;
        ret->nxt = ret->pre = NULL;

        Node* tmp = head->nxt;
        head->nxt = tmp->pre = ret;
        ret->pre = head;
        ret->nxt = tmp;

        return ret;
    }

    void update(Node *p, int value) {
        p->value = value;

        p->pre->nxt = p->nxt;
        p->nxt->pre = p->pre;

        p->pre = head;
        p->nxt = head->nxt;

        head->nxt->pre = p;
        head->nxt = p;
    }

    int delete_last() {
        Node *last = tail->pre;

        last->pre->nxt = last->nxt;
        last->nxt->pre = last->pre;

        int ret = last->key;
        delete last;

        return ret;
    }

};
```

---

## 152. [乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/description/?envType=study-plan-v2&envId=top-100-liked)

> 我不喜欢这个题目，出题人玩文字游戏。保证了答案不超过`int`，但是在数据中刻意让运算过程的数字爆掉了`long long`。与约定俗成的习惯不符。

维护`minDot`和`maxDot`两个变量，保存选择当前数字作为结尾的情况下，能获得的最大乘积和最小乘积分别是多少。

```cpp
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        long long maxDot = nums[0];
        long long minDot = nums[0];
        long long ans = nums[0];
        for (int i = 1; i < nums.size(); ++ i) {
            long long x = nums[i];
            if (x > 0) {
                ans = max(ans, max(maxDot * x, x));
                maxDot = max(x, maxDot * x);
                minDot = min((long long)x, minDot * x);
                if (minDot < -0x7fffffff) minDot = -0x7fffffff;
            } else {
                ans = max(ans, max(minDot * x, x));
                long long tmp = maxDot;
                maxDot = max(x, minDot * x);
                minDot = min((long long)x, tmp * x);
                if (minDot < -0x7fffffff) minDot = -0x7fffffff;
            }
        }
        return ans;
    }
};
```

---

## 155. [最小栈](https://leetcode.cn/problems/min-stack/description/?envType=study-plan-v2&envId=top-100-liked)

用了两个数组，一个数组维护栈，另一个数组维护单调栈。因为更靠前更小的数字可以阻止后面所有比他大的数字成为最小值。

```cpp
class MinStack {
    vector<int> sta;
    vector<int> minStack;
public:
    MinStack() {
        sta.clear();
        minStack.clear();
    }

    void push(int val) {
        sta.push_back(val);
        if(minStack.empty() || minStack.back() >= val)
            minStack.push_back(val);
    }

    void pop() {
        if (minStack.back() == sta.back()) {
            minStack.pop_back();
            sta.pop_back();
        } else sta.pop_back();
    }

    int top() {
        return sta.back();
    }

    int getMin() {
        return minStack.back();
    }
};
```

---

## 208. [实现 Trie](https://leetcode.cn/problems/implement-trie-prefix-tree/description/?envType=study-plan-v2&envId=top-100-liked)

字面意思，实现一个字典树。写了一个动态的。

```cpp
struct Node {
    Node* ch[26];
    bool flag;
};
class Trie {
    Node *rt;
public:
    Trie() {
        rt = newNode();
    }

    Node* newNode() {
        Node *ret = new Node();
        for (int i = 0;i < 26; ++ i) {
            ret->ch[i] = NULL;
        }
        ret->flag = false;
        return ret;
    }

    void insert(string word) {
        Node *nw = rt;
        for (int i = 0;i < word.size(); ++ i) {
            if (nw->ch[word[i] - 'a'] == NULL) {
                nw->ch[word[i] - 'a'] = newNode();
            }
            nw = nw->ch[word[i] - 'a'];
        }
        nw->flag = true;
    }

    bool search(string word) {
        Node *nw = rt;
        for (int i = 0;i < word.size(); ++ i) {
            if (nw->ch[word[i] - 'a'] == NULL) return false;
            nw = nw->ch[word[i] - 'a'];
        }
        return nw->flag;
    }

    bool startsWith(string prefix) {
        Node *nw = rt;
        for (int i = 0;i < prefix.size(); ++ i) {
            if (nw->ch[prefix[i] - 'a'] == NULL) return false;
            nw = nw->ch[prefix[i] - 'a'];
        }
        return true;
    }
};
```

---

## 230. [二叉搜索树中第K小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/?envType=study-plan-v2&envId=top-100-liked)

用记忆话的方式计算节点的size，然后在二叉树上二分

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    unordered_map<TreeNode*, int> siz;
    int get_size(TreeNode *root) {
        if (root == NULL) return 0;
        if (siz.contains(root)) return siz[root];
        siz[root] = 1;
        if (root->left) siz[root] += get_size(root->left);
        if (root->right) siz[root] += get_size(root->right);
        return siz[root];
    }
    int kthSmallest(TreeNode* root, int k) {
        if (get_size(root->left) + 1 == k) {
            return root->val;
        }
        if (get_size(root->left) < k) {
            k -= get_size(root->left) + 1;
            return kthSmallest(root->right, k);
        } else return kthSmallest(root->left, k);
    }
};
```

---

## 238. [移动零](https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

双指针，一个指针遍历数组，另一个指针指向下一个非零元素应该存放的位置。

```cpp
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int j = 0;
        for (int i = 0; i < nums.size(); ++ i) {
            if (nums[i] == 0) continue;
            nums[j++] = nums[i];
        }
        while(j < nums.size()) nums[j++] = 0;
    }
};
```

---

---

## 239. [滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/?envType=study-plan-v2&envId=top-100-liked)

单调栈，维护递减的单调栈。

```rust
use std::collections::VecDeque;

impl Solution {
    pub fn max_sliding_window(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut result: Vec<i32> = Vec::new();
        let mut sta: VecDeque<usize> = VecDeque::new();
        for i in 0..k-1 {
            while !sta.is_empty() && nums[i as usize] >= nums[*sta.back().unwrap()] {
                sta.pop_back();
            }
            sta.push_back(i as usize);
        }
        for i in k-1..nums.len() as i32{
            while !sta.is_empty() && nums[i as usize] >= nums[*sta.back().unwrap()] {
                sta.pop_back();
            }
            sta.push_back(i as usize);
            while i - *sta.front().unwrap() as i32 >= k {
                sta.pop_front();
            }
            result.push(nums[sta.front().unwrap().clone()])
        }
        result
    }
}
```

---

## 287. [寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)

很巧妙的思路，将数组中保存的数值看作连出的有向边，存在重复数字时一定存在环。快慢指针找到环之后，就可以将慢指针放到起点同步开始走，重合点就是入环点就是答案，可以列公式证明。

```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int x = 0, y = 0;
        do {
            x = nums[nums[x]];
            y = nums[y];
        } while(x != y);
        y = 0;
        while(x != y) {
            x = nums[x];
            y = nums[y];
        }
        return x;
    }
};
```

---

## 300. [最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/description/?envType=study-plan-v2&envId=top-100-liked)

树状数组加速动态规划即可，动态保存权值的前缀区间内最大的数字是多少。

```cpp
#define lowbit(x) (x&(-x))
class Solution {
    int c[2510], n;
public:
    void modify(int x, int y) {
        for (;x<=n;x+=lowbit(x))
            c[x] = max(c[x],y);
    }
    int query(int x) {
        int ret = 0;
        for(;x;x-=lowbit(x))
            ret = max(ret, c[x]);
        return ret;
    }
    int lengthOfLIS(vector<int>& nums) {
        vector<int> sorted_nums = nums;
        sort(sorted_nums.begin(), sorted_nums.end());
        int ans = 0;
        n = sorted_nums.size() + 2;
        memset(c, 0, sizeof c);
        for (int i = 0; i < nums.size(); ++ i) {
            int x = lower_bound(sorted_nums.begin(), sorted_nums.end(), nums[i]) - sorted_nums.begin() + 2;
            int fx = query(x-1) + 1;
            ans = max(ans, fx);
            modify(x, fx);
        }
        return ans;
    }
};
```

---

## 347. [前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/description/?envType=study-plan-v2&envId=top-100-liked)

用 HashMap 计算出每个数字出现的次数，然后用桶排序可以将复杂度控制到 $O(n)$

```cpp
class Solution {
    vector<int> ws[100010];
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> hash;
        for (int i = 0; i < nums.size(); ++ i) {
            if (hash.contains(nums[i])) {
                hash[nums[i]] += 1;
            } else {
                hash[nums[i]] = 1;
            }
        }
        for (int i = 0; i < nums.size(); ++i) ws[i].clear();
        for (auto it = hash.begin(); it != hash.end(); ++ it) {
            ws[it->second].push_back(it->first);
        }
        vector<int> ans;
        for (int i = nums.size(); i > 0; -- i) {
            if (ws[i].size() != 0) {
                for (int x : ws[i]) {
                    ans.push_back(x);
                    if (-- k == 0) break;
                }
            }
            if(k == 0) break;
        }
        return ans;
    }
};
```
