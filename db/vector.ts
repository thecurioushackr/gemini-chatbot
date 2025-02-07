import { customType } from "drizzle-orm/pg-core";

export const vector = customType<{ data: number[]; notNull: boolean; default: boolean }>({
  dataType() {
    return "vector";
  },
  toDriver(value: number[]): string {
    return `[${value.join(",")}]`;
  },
  fromDriver(value: unknown): number[] {
    if (typeof value !== 'string') {
      throw new Error('Vector value must be a string');
    }
    return JSON.parse(value);
  },
});
