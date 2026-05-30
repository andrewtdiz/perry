const s: string = "ababa";

function typedIncludes(search: string, position?: number) {
  if (position === undefined) {
    return s.includes(search);
  }
  return s.includes(search, position);
}

const dynamicReceiver: any = s;
function dynamicIncludes(search: string, position?: number) {
  if (position === undefined) {
    return dynamicReceiver.includes(search);
  }
  return dynamicReceiver.includes(search, position);
}

const cases: Array<[string, string, number?]> = [
  ["default", "a"],
  ["from 1", "a", 1],
  ["from 2", "a", 2],
  ["from 5", "a", 5],
  ["negative", "a", -10],
  ["nan", "a", NaN],
  ["infinity", "a", Infinity],
  ["fraction", "b", 2.9],
  ["empty infinity", "", Infinity],
];

for (const [label, search, position] of cases) {
  console.log("typed", label, typedIncludes(search, position));
}

for (const [label, search, position] of cases) {
  console.log("dynamic", label, dynamicIncludes(search, position));
}
