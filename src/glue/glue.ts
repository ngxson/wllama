import {
  GLUE_MESSAGE_PROTOTYPES,
  GLUE_VERSION,
  type GlueMsg,
} from './messages';

/**
 * Glue is a simple binary protocol for serializing and deserializing messages.
 * It is inspired by protobuf, but much simpler.
 *
 * Interested in extending Glue? Open an issue on GitHub!
 */

type GlueType =
  | 'str'
  | 'int'
  | 'float'
  | 'bool'
  | 'raw'
  | 'arr_str'
  | 'arr_int'
  | 'arr_float'
  | 'arr_bool'
  | 'arr_raw'
  | 'null';

const GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);

export interface GlueField {
  type: GlueType;
  name: string;
  isNullable: boolean;
}

export interface GlueMessageProto {
  name: string;
  structName: string;
  className: string;
  fields: GlueField[];
}

const GLUE_DTYPE_NULL = 0;
const GLUE_DTYPE_BOOL = 1;
const GLUE_DTYPE_INT = 2;
const GLUE_DTYPE_FLOAT = 3;
const GLUE_DTYPE_STRING = 4;
const GLUE_DTYPE_RAW = 5;
const GLUE_DTYPE_ARRAY_BOOL = 6;
const GLUE_DTYPE_ARRAY_INT = 7;
const GLUE_DTYPE_ARRAY_FLOAT = 8;
const GLUE_DTYPE_ARRAY_STRING = 9;
const GLUE_DTYPE_ARRAY_RAW = 10;

const TYPE_MAP: Record<GlueType, number> = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL,
};

export function glueDeserialize(buf: Uint8Array): GlueMsg {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen?: number) => {
    const length = customLen ?? readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem: () => any) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;

  const readField = (field: GlueField) => {
    switch (field.type) {
      case 'str':
        return readString();
      case 'int':
        return readInt32();
      case 'float':
        return readFloat();
      case 'bool':
        return readBool();
      case 'raw':
        return readRaw();
      case 'arr_str':
        return readArray(readString);
      case 'arr_int':
        return readArray(readInt32);
      case 'arr_float':
        return readArray(readFloat);
      case 'arr_bool':
        return readArray(readBool);
      case 'arr_raw':
        return readArray(readRaw);
      case 'null':
        return readNull();
    }
  };

  const magicValid =
    buf[0] === GLUE_MAGIC[0] &&
    buf[1] === GLUE_MAGIC[1] &&
    buf[2] === GLUE_MAGIC[2] &&
    buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error('Invalid magic number');
  }

  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error('Invalid version number');
  }

  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }

  const output: any = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }

  return output;
}

export function glueSerialize(msg: GlueMsg): Uint8Array {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }

  const bufs: Uint8Array[] = [];

  const writeUint32 = (value: number) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value: number) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value: number) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value: boolean) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value: string) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value: Uint8Array) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value: any[], writeItem: (item: any) => void) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {};

  //////////////////

  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    // write proto ID
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = (msg as any)[field.name];
    if (!field.isNullable && (val === null || val === undefined)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === undefined) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case 'str':
        writeString(val);
        break;
      case 'int':
        writeInt32(val);
        break;
      case 'float':
        writeFloat(val);
        break;
      case 'bool':
        writeBool(val);
        break;
      case 'raw':
        writeRaw(val);
        break;
      case 'arr_str':
        writeArray(val, writeString);
        break;
      case 'arr_int':
        writeArray(val, writeInt32);
        break;
      case 'arr_float':
        writeArray(val, writeFloat);
        break;
      case 'arr_bool':
        writeArray(val, writeBool);
        break;
      case 'arr_raw':
        writeArray(val, writeRaw);
        break;
      case 'null':
        writeNull();
        break;
    }
  }

  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}
