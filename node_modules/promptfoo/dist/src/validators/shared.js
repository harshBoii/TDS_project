"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NunjucksFilterMapSchema = void 0;
const zod_1 = require("zod");
exports.NunjucksFilterMapSchema = zod_1.z.record(zod_1.z.string(), zod_1.z.function(zod_1.z.tuple([zod_1.z.any()]).rest(zod_1.z.any()), zod_1.z.string()));
//# sourceMappingURL=shared.js.map