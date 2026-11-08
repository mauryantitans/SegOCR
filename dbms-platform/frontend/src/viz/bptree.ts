/** A real B+ tree (order = max children per node) with event logging for visualization. */
export interface BNode { id: number; leaf: boolean; keys: number[]; children: BNode[]; next: BNode | null }
export interface Snap { id: number; leaf: boolean; keys: number[]; children: Snap[] }

export class BPlusTree {
  root: BNode;
  private seq = 1;
  constructor(public order: number) {
    if (order < 3) throw new Error("order must be >= 3");
    this.root = this.node(true);
  }
  private node(leaf: boolean): BNode { return { id: this.seq++, leaf, keys: [], children: [], next: null }; }
  private get maxKeys() { return this.order - 1; }

  private childIndex(n: BNode, key: number) {
    let i = 0;
    while (i < n.keys.length && key >= n.keys[i]) i++;
    return i;
  }

  /** Insert a key; returns human-readable events describing what happened. */
  insert(key: number): string[] {
    const events: string[] = [];
    const path: BNode[] = [];
    let n = this.root;
    while (!n.leaf) { path.push(n); n = n.children[this.childIndex(n, key)]; }
    if (n.keys.includes(key)) return [`${key} is already in the tree (unique index): no change`];
    const pos = n.keys.findIndex((k) => k > key);
    n.keys.splice(pos < 0 ? n.keys.length : pos, 0, key);
    events.push(`Inserted ${key} into leaf [${n.keys.join(", ")}]`);
    if (n.keys.length <= this.maxKeys) return events;

    // split leaf: right half moves to a new leaf, its first key is COPIED up
    const right = this.node(true);
    const mid = Math.ceil(n.keys.length / 2);
    right.keys = n.keys.splice(mid);
    right.next = n.next; n.next = right;
    let sep = right.keys[0];
    let newChild: BNode = right;
    events.push(`Leaf overflow (${this.order} keys > max ${this.maxKeys}): split into [${n.keys.join(", ")}] and [${right.keys.join(", ")}]; copy ${sep} up`);
    let left = n;

    while (true) {
      const parent = path.pop();
      if (!parent) {
        const root = this.node(false);
        root.keys = [sep]; root.children = [left, newChild];
        this.root = root;
        events.push(`Root split: new root [${sep}] — the tree grew one level`);
        return events;
      }
      const idx = parent.children.indexOf(left);
      parent.keys.splice(idx, 0, sep);
      parent.children.splice(idx + 1, 0, newChild);
      if (parent.keys.length <= this.maxKeys) { events.push(`Separator ${sep} inserted into internal node [${parent.keys.join(", ")}]`); return events; }
      // split internal node: middle key MOVES up
      const m = Math.floor(parent.keys.length / 2);
      const up = parent.keys[m];
      const r = this.node(false);
      r.keys = parent.keys.splice(m + 1);
      parent.keys.splice(m, 1);
      r.children = parent.children.splice(m + 1);
      events.push(`Internal overflow: split into [${parent.keys.join(", ")}] and [${r.keys.join(", ")}]; move ${up} up`);
      left = parent; newChild = r; sep = up;
    }
  }

  search(key: number): { path: number[]; found: boolean; steps: string[] } {
    const path: number[] = []; const steps: string[] = [];
    let n = this.root;
    while (!n.leaf) {
      path.push(n.id);
      const i = this.childIndex(n, key);
      steps.push(`Internal [${n.keys.join(", ")}]: ${key} ${i === 0 ? `< ${n.keys[0]}` : i === n.keys.length ? `≥ ${n.keys[n.keys.length - 1]}` : `in [${n.keys[i - 1]}, ${n.keys[i]})`} → child ${i + 1}`);
      n = n.children[i];
    }
    path.push(n.id);
    const found = n.keys.includes(key);
    steps.push(`Leaf [${n.keys.join(", ")}]: ${found ? `found ${key}` : `${key} not present`}`);
    return { path, found, steps };
  }

  /** Range scan lo ≤ k ≤ hi: descend once, then follow leaf links. */
  range(lo: number, hi: number): { path: number[]; leaves: number[]; keys: number[] } {
    const { path } = this.search(lo);
    let n: BNode | null = this.root;
    while (n && !n.leaf) n = n.children[this.childIndex(n, lo)];
    const leaves: number[] = []; const keys: number[] = [];
    while (n) {
      leaves.push(n.id);
      for (const k of n.keys) if (k >= lo && k <= hi) keys.push(k);
      if (n.keys.length && n.keys[n.keys.length - 1] > hi) break;
      n = n.next;
    }
    return { path, leaves, keys };
  }

  height(): number { let h = 1; let n = this.root; while (!n.leaf) { h++; n = n.children[0]; } return h; }

  snapshot(): Snap {
    const s = (n: BNode): Snap => ({ id: n.id, leaf: n.leaf, keys: [...n.keys], children: n.children.map(s) });
    return s(this.root);
  }

  /** In-order keys from the leaf chain (used by tests to verify invariants). */
  leafKeys(): number[] {
    let n: BNode = this.root;
    while (!n.leaf) n = n.children[0];
    const out: number[] = [];
    for (let c: BNode | null = n; c; c = c.next) out.push(...c.keys);
    return out;
  }
}
