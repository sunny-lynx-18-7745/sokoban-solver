// Push-optimal Sokoban solver, room by room.
//
// Approach: the generator's levels are a 1-wide hallway with 1-4 rooms off it,
// each room holding 3-4 boxes and as many goals, and a box pushed out of its room
// can never come back — so a level's push-optimum is the sum of its rooms'
// optima. Each room is solved with a breadth-first search over pushes (a state is
// the boxes' squares plus the player's reachable region, named by its lowest
// square), which makes the first solved state push-optimal. Two deadlocks are
// pruned: dead squares (a box there can never reach a goal, found by pulling back
// from the goals) and frozen 2x2 blocks of walls and boxes with a box off its
// goal. Neither prunes a solvable state, so the optimum stands. The search is
// bounded by work (250k states), never the clock, so two runs agree byte for byte.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MAX_STATES = 250_000;
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

type Level = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares from which a box alone can be pushed onto some goal (pulled back from the goals). */
function liveSquares(L: Level): Uint8Array {
  const { w, wall, goals } = L;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Flood-fill the player's region from `from` with `occ` in the way; returns its lowest square. */
function reach(L: Level, occ: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { w, wall } = L;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occ[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(L: Level, occ: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = L;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goal[a]) || (occ[b] && !goal[b]) || (occ[c] && !goal[c]) || (occ[d] && !goal[d])) return true;
  }
  return false;
}

/** Shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(L: Level, occ: Uint8Array, from: number, to: number): string {
  const { w, wall } = L;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** Push-optimal solution of a level: {pushes, moves} (moves in LURD, upper = push), or null. */
function solvePushOptimal(L: Level, maxStates = MAX_STATES): { pushes: number; moves: string } | null {
  const { w, wall, goals } = L;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(L);
  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  if (L.boxes.some((b) => !live[b])) return null;
  const small = L.boxes.length <= 5; // numeric key is exact while size ** (boxes + 1) < 2 ** 53
  const keyOf = (bs: number[], low: number): number | string => {
    if (!small) return `${low}:${bs.join(',')}`;
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };
  const boxesList: number[][] = [L.boxes];
  const playerList: number[] = [L.player];
  const parent: number[] = [-1];
  const index = new Map<number | string, number>();
  for (const b of L.boxes) occ[b] = 1;
  index.set(keyOf(L.boxes, reach(L, occ, L.player, seen, ++stamp, stack)), 0);
  for (const b of L.boxes) occ[b] = 0;
  let found = solved(L.boxes) ? 0 : -1;
  for (let head = 0; found < 0 && head < boxesList.length; head++) {
    const boxes = boxesList[head]!;
    for (const b of boxes) occ[b] = 1;
    reach(L, occ, playerList[head]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0;
        occ[t] = 1;
        const frozen = frozenBlock(L, occ, goal, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          key = keyOf(next, reach(L, occ, b, mark, ++stamp, stack));
        }
        occ[t] = 0;
        occ[b] = 1;
        if (frozen || index.has(key)) continue;
        if (boxesList.length >= maxStates) {
          for (const x of boxes) occ[x] = 0;
          return null;
        }
        index.set(key, boxesList.length);
        boxesList.push(next!);
        playerList.push(b);
        parent.push(head);
        if (solved(next!)) {
          found = boxesList.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occ[x] = 0;
  }
  if (found < 0) return null;
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = L.player;
  let boxes = L.boxes;
  for (const s of path) {
    const nextBoxes = boxesList[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occ[b] = 1;
    moves += walkTo(L, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

type Room = { squares: number[]; boxes: number[]; goals: number[]; done?: boolean };

/** The level's rooms: cut the dead squares of the player's row (the hallway) out of the floor. */
function roomsOf(L: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = L;
  const live = liveSquares(L);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve room by room: other rooms' boxes stand as walls (on their goals once done). */
function solveByRooms(L: Level, maxStates = MAX_STATES): { pushes: number; moves: string } | null {
  const rooms = roomsOf(L);
  if (rooms === null) return solvePushOptimal(L, maxStates);
  let player = L.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = L.wall.slice();
    for (const other of rooms)
      if (other !== room) for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    const sub: Level = {
      w: L.w,
      h: L.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushOptimal(sub, maxStates);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    let p = player;
    for (const ch of solved.moves.toLowerCase()) p += ch === 'u' ? -L.w : ch === 'd' ? L.w : ch === 'l' ? -1 : 1;
    player = p;
    room.done = true;
  }
  return { pushes, moves };
}

for (const id of readdirSync(I).sort()) {
  const level = parse(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  const solved = solveByRooms(level);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${solved?.moves ?? ''}\n`);
}
