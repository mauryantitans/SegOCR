import { describe, expect, it } from "vitest";
import { BPlusTree } from "./bptree";
import { analyze, candidateKeys, closure, decomposeBCNF, minimalCover, parseFDs, preservesDependencies, synthesize3NF } from "./fd";

describe("B+ tree", () => {
  it("keeps keys sorted in the leaf chain and stays balanced", () => {
    const t = new BPlusTree(4);
    const keys = [50, 10, 70, 20, 30, 60, 80, 90, 40, 15, 25, 35, 65, 75, 85, 95, 5, 1];
    keys.forEach((k) => t.insert(k));
    expect(t.leafKeys()).toEqual([...keys].sort((a, b) => a - b));
    const depths = new Set<number>();
    const walk = (n: ReturnType<BPlusTree["snapshot"]>, d: number) => (n.leaf ? depths.add(d) : n.children.forEach((c) => walk(c, d + 1)));
    walk(t.snapshot(), 1);
    expect(depths.size).toBe(1);
    const check = (n: ReturnType<BPlusTree["snapshot"]>) => { expect(n.keys.length).toBeLessThanOrEqual(3); if (!n.leaf) { expect(n.children.length).toBe(n.keys.length + 1); n.children.forEach(check); } };
    check(t.snapshot());
  });
  it("finds keys, ranges and rejects duplicates", () => {
    const t = new BPlusTree(3);
    for (let i = 1; i <= 20; i++) t.insert(i * 10);
    expect(t.search(70).found).toBe(true);
    expect(t.search(75).found).toBe(false);
    expect(t.range(45, 105).keys).toEqual([50, 60, 70, 80, 90, 100]);
    expect(t.insert(70)[0]).toMatch(/already/);
    expect(t.height()).toBeGreaterThan(2);
  });
});

describe("functional dependencies", () => {
  const attrs = ["A", "B", "C", "D", "E"];
  const fds = parseFDs(["A -> B", "B -> C", "C, D -> E"], attrs);
  it("computes closures and keys", () => {
    expect([...closure(["A"], fds)].sort()).toEqual(["A", "B", "C"]);
    expect(candidateKeys(attrs, fds)).toEqual([["A", "D"]]);
  });
  it("classifies normal forms", () => {
    expect(analyze(attrs, fds).nf).toBe("1NF"); // A -> B is partial on key AD
    const r = ["S", "C", "T"]; // classic: S,C -> T ; T -> C  (3NF but not BCNF)
    const f2 = parseFDs(["S, C -> T", "T -> C"], r);
    const a = analyze(r, f2);
    expect(a.keys).toEqual([["C", "S"], ["S", "T"]]);
    expect(a.nf).toBe("3NF");
    const bc = decomposeBCNF(r, f2);
    expect(bc.relations.length).toBe(2);
    expect(preservesDependencies(bc.relations, f2).lost.length).toBe(1);
  });
  it("builds a minimal cover and 3NF synthesis", () => {
    const f = parseFDs(["A -> B, C", "B -> C", "A, B -> C"], ["A", "B", "C"]);
    expect(minimalCover(f).map((x) => `${x.lhs.join("")}->${x.rhs.join("")}`).sort()).toEqual(["A->B", "B->C"]);
    expect(synthesize3NF(["A", "B", "C"], f).map((r) => r.join(""))).toEqual(["AB", "BC"]);
  });
});
