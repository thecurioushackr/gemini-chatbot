import { Message } from "ai";
import { InferSelectModel } from "drizzle-orm";
import {
  pgTable,
  varchar,
  timestamp,
  json,
  uuid,
  boolean,
  text,
  real,
} from "drizzle-orm/pg-core";

import { vector } from "./vector";

export const user = pgTable("User", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  email: varchar("email", { length: 64 }).notNull(),
  password: varchar("password", { length: 64 }),
});

export type User = InferSelectModel<typeof user>;

export const chat = pgTable("Chat", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  createdAt: timestamp("createdAt").notNull(),
  messages: json("messages").notNull(),
  userId: uuid("userId")
    .notNull()
    .references(() => user.id),
});

export type Chat = Omit<InferSelectModel<typeof chat>, "messages"> & {
  messages: Array<Message>;
};

export const reservation = pgTable("Reservation", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  createdAt: timestamp("createdAt").notNull(),
  details: json("details").notNull(),
  hasCompletedPayment: boolean("hasCompletedPayment").notNull().default(false),
  userId: uuid("userId")
    .notNull()
    .references(() => user.id),
});

export type Reservation = InferSelectModel<typeof reservation>;

// Memory System Tables
export const memory = pgTable("Memory", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  content: text("content").notNull(),
  timestamp: timestamp("timestamp").notNull(),
  type: varchar("type", { length: 16 }).notNull(), // 'episodic' or 'semantic'
  importance: real("importance").notNull(),
  embedding: vector("embedding", { dimensions: 768 }), // Gemini embedding dimension
  userId: uuid("userId")
    .notNull()
    .references(() => user.id),
});

export type Memory = InferSelectModel<typeof memory>;
