#include "finalize-omp.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Semantics/tools.h"

#include <map>
#include <variant>

namespace Fortran::semantics {

using namespace parser::literals;

namespace {
using Subprograms =
    std::variant<parser::SubroutineSubprogram *, parser::FunctionSubprogram *>;

parser::Name *getNameFromVariant(Subprograms &x) {
  if (std::holds_alternative<parser::FunctionSubprogram *>(x)) {
    parser::FunctionStmt &Stmt =
        std::get<0>(std::get<parser::FunctionSubprogram *>(x)->t).statement;
    return &std::get<parser::Name>(Stmt.t);
  }

  if (std::holds_alternative<parser::SubroutineSubprogram *>(x)) {
    parser::SubroutineStmt &Stmt =
        std::get<0>(std::get<parser::SubroutineSubprogram *>(x)->t).statement;
    return &std::get<parser::Name>(Stmt.t);
  }

  return nullptr;
}

class GatherCallRefs {
public:
  GatherCallRefs(std::map<std::string, Subprograms> &subprograms)
      : subprograms(subprograms) {}

  // Default action for a parse tree node is to visit children.
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

  void Post(parser::Call &call) {
    if (std::holds_alternative<parser::Name>(std::get<0>(call.t).u))
      callNames.push_back(std::get<parser::Name>(std::get<0>(call.t).u));
  }

  void Post(parser::CallStmt &call) {
    if (call.typedCall) {
      // find function in list of functions, get name from function
      auto subprogram = subprograms.find(call.typedCall->proc().GetName());
      if (subprogram != subprograms.end())
        callNames.push_back(*getNameFromVariant(subprogram->second));
    }
  }

  std::map<std::string, Subprograms> &subprograms;
  llvm::SmallVector<parser::Name> callNames;
};

class GatherDeclareTargetConstruct {
public:
  GatherDeclareTargetConstruct() {}

  // Default action for a parse tree node is to visit children.
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

  void Post(parser::OpenMPDeclareTargetConstruct &DeclTar) {
    declTarCons.push_back(&DeclTar);
  }

  llvm::SmallVector<parser::OpenMPDeclareTargetConstruct *> declTarCons;
};

} // namespace

// This pass works by passing over the modules PFT and collecting all of the
// ProgramUnits(subroutines / functions) and InterfaceBlocks related to these
// units as it passes through these nodes. When it meets a declare target
// or target region node, it begins to parse through the region or function tied
// to the declare target or target region. it reads through the extended-list of
// the declare target directive's clause finding functions (the current function
// should be referred to in this declare target if a list is specified), if no
// list is specified the function the declare target is in, is considered the
// only function in the declare target list. For target region's the functions
// calls contained within the region are recursively processed.

// The functions processed with the algorithm then have a declare target
// construct generated within them, referencing themselves with the device_type
// Any (as even if it's from a nohost region we cannot assume it's not used
// elsewhere for the host without more information, we can only state it is
// needed for device), only if they do not already have one already. If there
// is one in existence, and the device_type is host then we switch the device
// type to any, as it now exists on device. However, if the declare target has
// more than just the function within the declare target's to list, then we
// seperate it out into a new declare target and change the device_type of this
// newly generated declare target, the original declare target remains the same
// minus the removed function name. This is intended to avoid issues of
// switching device_type of unrelated declare target items. Declare target's
// with extended-lists or no list at all are ignored, as they already refer to
// the function they are nested in by default.

// Multiple declare targets referencing the same thing utilisng
// To clauses are not illegal (only when it's a link clause, which cannot be
// used in conjunction with functions), so on the off chance the pass generates
// more than one declare target with differing device_types (perhaps another
// declare target already refers to it and we did not find it as it's external)
// we handle this apon lowering, simply ignoring it if we already have a declare
// target with the same device_type, or changing it to any if we meet two of
// differing types referencing the same function.
class ImplicitDeclareTargetCapture {
  Subprograms currentSubprogram_;
  std::map<std::string, Subprograms> subPrograms_;
  std::map<std::string, parser::InterfaceBody *> interfaces_;
  bool insideOfInterfaceBody_ = false;

  parser::Messages &messages_;

  template <typename T>
  parser::OpenMPDeclareTargetConstruct *getDeclTarConstruct(
      T &body, parser::Name name) {

    GatherDeclareTargetConstruct gatherer{};
    parser::Walk(body, gatherer);
    if (gatherer.declTarCons.size() > 0) {
      // Find if there is a construct the function is already referenced in
      for (auto *declTar : gatherer.declTarCons) {
        if (existsInDeclareTargetOrClauseless(*declTar, name))
          return declTar;
      }
    }

    return nullptr;
  }

  std::list<parser::OpenMPDeclarativeConstruct> *getDeclarativeConstructList(
      parser::InterfaceBody &body) {
    if (auto *func{parser::Unwrap<parser::InterfaceBody::Function>(body.u)}) {
      auto &spec{
          std::get<common::Indirection<parser::SpecificationPart>>(func->t)};
      return &std::get<std::list<parser::OpenMPDeclarativeConstruct>>(
          spec.value().t);
    }

    if (auto *subr{parser::Unwrap<parser::InterfaceBody::Subroutine>(body.u)}) {
      auto &spec{
          std::get<common::Indirection<parser::SpecificationPart>>(subr->t)};
      return &std::get<std::list<parser::OpenMPDeclarativeConstruct>>(
          spec.value().t);
    }

    return nullptr;
  }

  void createOrModifyDeclareTargetTo(parser::OpenMPDeclareTargetConstruct *x,
      std::list<parser::OpenMPDeclarativeConstruct> *conList, parser::Name name,
      parser::OmpClause::DeviceType *parentsDeviceTypeClause) {
    auto createDeclareTargetTo = [](parser::Name name) {
      // Create the object list containing the functions name
      std::list<parser::OmpObject> objects;
      objects.push_back(parser::OmpObject{
          parser::Designator{parser::DataRef{std::move(name)}}});
      auto objList = parser::OmpObjectList{std::move(objects)};
      // Create ClauseList of: to(FunctionName) device_type(Any)
      std::list<parser::OmpClause> clauses;
      clauses.push_back(parser::OmpClause::To{std::move(objList)});
      clauses.push_back(parser::OmpClause::DeviceType{
          Fortran::parser::OmpDeviceTypeClause::Type::Any});
      auto newClauseList = parser::OmpClauseList{std::move(clauses)};
      // Create the Declare Target construct adding our clauses: declare target
      // to(FunctionName) device_type(Any)
      auto declTarget =
          parser::OmpDeclareTargetWithClause{std::move(newClauseList)};
      auto specifier = parser::OmpDeclareTargetSpecifier{std::move(declTarget)};
      return parser::OpenMPDeclareTargetConstruct{
          parser::Verbatim{}, std::move(specifier)};
    };

    if (x) {
      auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x->t)};
      auto *clauseList{parser::Unwrap<parser::OmpClauseList>(spec.u)};
      parser::OmpClause::DeviceType *deviceType = nullptr;

      // clauseless and extended-list cases don't need to be considered here,
      // they should just return with no alterations. This is because
      // extended-lists cannot have device_types and are implicitly device_type
      // any already, and declare target without any clauses or extended-lists
      // are implicitly extended-lists.
      if (clauseList && !clauseList->v.empty()) {
        for (auto &clause : clauseList->v) {
          if (auto *devType{
                  std::get_if<parser::OmpClause::DeviceType>(&clause.u)}) {
            deviceType = devType;
          }
        }

        // We have a declare target already, however, it is marked host and the
        // function is sharing it with other list-items, we do not want to
        // change the markings on the existing declare target, so we wish to
        // split our function from the rest of the list into a newly generated
        // declare target.
        for (auto &clause : clauseList->v) {
          if (auto *toClause{std::get_if<parser::OmpClause::To>(&clause.u)}) {
            if (toClause->v.v.size() > 1 &&
                deviceType->v.v ==
                    Fortran::parser::OmpDeviceTypeClause::Type::Host) {
              // remove from existing declare target with multiple objects
              for (auto ompObject = toClause->v.v.begin();
                   ompObject != toClause->v.v.end(); ompObject++) {
                if (auto *objName{parser::Unwrap<parser::Name>(ompObject->u)})
                  if (objName->ToString() == name.ToString()) {
                    toClause->v.v.erase(ompObject);
                    break;
                  }
              }

              // generate the new declare target
              conList->push_back(createDeclareTargetTo(name));
              return;
            }
          }
        }

        // This function, is required by a target function, we must swap
        // it to a new device type.
        if ((parentsDeviceTypeClause &&
                parentsDeviceTypeClause->v.v != deviceType->v.v) ||
            deviceType->v.v ==
                Fortran::parser::OmpDeviceTypeClause::Type::Host) {
          deviceType->v.v = Fortran::parser::OmpDeviceTypeClause::Type::Any;
        }
      }
      return;
    }

    // There is no declare target referencing this function within the function
    // yet, we must generate one.
    conList->push_back(createDeclareTargetTo(name));
  }

  bool isTargetRegion(parser::OpenMPBlockConstruct &x) {
    auto &blockDir{std::get<parser::OmpBlockDirective>(
        std::get<parser::OmpBeginBlockDirective>(x.t).t)};
    if (blockDir.v == llvm::omp::Directive::OMPD_target)
      return true;
    return false;
  }

  bool existsInDeclareTargetOrClauseless(
      parser::OpenMPDeclareTargetConstruct &x, parser::Name name) {
    auto existsInObjList = [](parser::OmpObjectList &objList,
                               parser::Name name) {
      for (auto &ompObject : objList.v)
        if (auto *objName{parser::Unwrap<parser::Name>(ompObject)})
          if (objName->ToString() == name.ToString())
            return true;
      return false;
    };

    auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x.t)};
    if (parser::OmpObjectList *
        objectList{parser::Unwrap<parser::OmpObjectList>(spec.u)}) {
      return existsInObjList(*objectList, name);
    }

    if (auto *clauseList{parser::Unwrap<parser::OmpClauseList>(spec.u)}) {
      // declare target with no clause case.
      if (clauseList->v.empty())
        return true;

      for (auto &clause : clauseList->v) {
        if (auto *toClause{std::get_if<parser::OmpClause::To>(&clause.u)}) {
          return existsInObjList(toClause->v, name);
        }

        if (auto *linkClause{std::get_if<parser::OmpClause::Link>(&clause.u)}) {
          return existsInObjList(linkClause->v, name);
        }
      }
    }

    return false;
  }

  void addDeclareTarget(Subprograms &x, parser::Name name,
      parser::OmpClause::DeviceType *parentsDeviceTypeClause) {
    std::list<parser::OpenMPDeclarativeConstruct> *conList = nullptr;
    if (std::holds_alternative<parser::FunctionSubprogram *>(x)) {
      conList = &std::get<std::list<parser::OpenMPDeclarativeConstruct>>(
          std::get<parser::SpecificationPart>(
              std::get<parser::FunctionSubprogram *>(x)->t)
              .t);
    } else if (std::holds_alternative<parser::SubroutineSubprogram *>(x)) {
      conList = &std::get<std::list<parser::OpenMPDeclarativeConstruct>>(
          std::get<parser::SpecificationPart>(
              std::get<parser::SubroutineSubprogram *>(x)->t)
              .t);
    }

    GatherDeclareTargetConstruct gatherer{};
    parser::OpenMPDeclareTargetConstruct *declareTarget = nullptr;
    if (std::holds_alternative<parser::SubroutineSubprogram *>(x))
      declareTarget = getDeclTarConstruct(
          *std::get<parser::SubroutineSubprogram *>(x), name);
    else
      declareTarget =
          getDeclTarConstruct(*std::get<parser::FunctionSubprogram *>(x), name);

    createOrModifyDeclareTargetTo(
        declareTarget, conList, name, parentsDeviceTypeClause);

    // If the declare targetted function has an interface we must make sure
    // the changes are reflected in the interface, as they should be identical
    // executing the same function on it, should result in the same set of
    // changes, if they are not idenitcal, then this is already incorrect
    // OpenMP.
    auto interface = interfaces_.find(name.ToString());
    if (interface != interfaces_.end()) {
      createOrModifyDeclareTargetTo(
          getDeclTarConstruct(*interface->second, name),
          getDeclarativeConstructList(*interface->second), name,
          parentsDeviceTypeClause);
    }
  }

  void markDeclTarForEachProgramUnit(
      std::map<std::string, Subprograms> subPrograms, parser::Name name,
      parser::OmpClause::DeviceType *parentsDeviceTypeClause = nullptr,
      bool isOriginalDeclTar = false) {

    auto subProgram = subPrograms.extract(name.ToString());
    if (subProgram.empty())
      return;

    // Add a declare target to the subprogram and its InterfaceBlock (if
    // it has one) if one does not already exist
    if (!isOriginalDeclTar)
      addDeclareTarget(subProgram.mapped(), name, parentsDeviceTypeClause);

    GatherCallRefs gatherer{subPrograms_};
    if (std::holds_alternative<parser::SubroutineSubprogram *>(
            subProgram.mapped()))
      parser::Walk(
          *std::get<parser::SubroutineSubprogram *>(subProgram.mapped()),
          gatherer);
    else
      parser::Walk(*std::get<parser::FunctionSubprogram *>(subProgram.mapped()),
          gatherer);

    // Recurse on all calls found within the function, removing
    // visited functions from the programUnits list as we go, if it
    // is no longer in the list of units, then we have already
    // processed it and made it declare target
    for (auto v : gatherer.callNames) {
      markDeclTarForEachProgramUnit(subPrograms, v, parentsDeviceTypeClause);
    }
  }

public:
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}
  ImplicitDeclareTargetCapture(SemanticsContext &context)
      : messages_{context.messages()} {}

  // The entry point for rewriting declare target regions which
  // contain functions nested within to be declare target
  // themselves.
  void Post(parser::OpenMPBlockConstruct &x) {
    if (isTargetRegion(x)) {
      GatherCallRefs gatherer{subPrograms_};
      parser::Walk(std::get<parser::Block>(x.t), gatherer);

      auto deviceType = parser::OmpClause::DeviceType{
          Fortran::parser::OmpDeviceTypeClause::Type::Nohost};
      for (auto call : gatherer.callNames)
        markDeclTarForEachProgramUnit(subPrograms_, call, &deviceType);
    }
  }

  // The entry point for rewriting declare target specifiers which
  // contain functions nested within to be declare target themselves.
  void Post(parser::OpenMPDeclareTargetConstruct &x) {
    // InterfaceBody specifications of functions are handled as we parse the
    // functions, if we alter a function to add a declare target,
    // we check if an interface body also exists for it, if it does we must
    // keep them in synch. The OpenMP specification states that a user must
    // define synchronized interface and functions definitions for declare
    // target.
    if (!insideOfInterfaceBody_) {
      auto &spec{std::get<parser::OmpDeclareTargetSpecifier>(x.t)};
      if (parser::OmpObjectList *
          objectList{parser::Unwrap<parser::OmpObjectList>(spec.u)}) {
        for (auto &ompObject : objectList->v)
          if (auto *name{parser::Unwrap<parser::Name>(ompObject)})
            markDeclTarForEachProgramUnit(subPrograms_, *name, nullptr, true);
      } else if (auto *clauseList{
                     parser::Unwrap<parser::OmpClauseList>(spec.u)}) {
        parser::OmpClause::DeviceType *devType = nullptr;
        for (auto &clause : clauseList->v) {
          if (auto *deviceType{
                  std::get_if<parser::OmpClause::DeviceType>(&clause.u)}) {
            devType = deviceType;
          }
        }

        // We do not process link clauses these are not allowed to have
        // functions as list items. And we do not care about host specific
        // declare targets, the specification states that the implicit marking
        // only happens for nohost (and by extension any, one would assume).
        // Every non-declare target function can be assumed to be "host".
        if (!devType ||
            devType->v.v != Fortran::parser::OmpDeviceTypeClause::Type::Host)
          for (auto &clause : clauseList->v) {
            if (auto *toClause{std::get_if<parser::OmpClause::To>(&clause.u)}) {
              for (auto &ompObject : toClause->v.v)
                if (auto *name{parser::Unwrap<parser::Name>(ompObject)})
                  markDeclTarForEachProgramUnit(
                      subPrograms_, *name, devType, true);
            }
          }

        // The default "declare target" inside of a function case, we must
        // process the functions called from within
        if (clauseList->v.empty()) {
          if (auto *name = getNameFromVariant(currentSubprogram_)) {
            markDeclTarForEachProgramUnit(subPrograms_, *name, nullptr, true);
          }
        }
      }
    }
  }

  bool Pre(parser::FunctionSubprogram &x) {
    insideOfInterfaceBody_ = false;
    auto &stmt =
        std::get<parser::Statement<parser::FunctionStmt>>(x.t).statement;
    auto name = std::get<parser::Name>(stmt.t);
    subPrograms_[name.ToString()] = &x;
    currentSubprogram_ = &x;
    return true;
  }

  bool Pre(parser::SubroutineSubprogram &x) {
    insideOfInterfaceBody_ = false;
    auto &stmt =
        std::get<parser::Statement<parser::SubroutineStmt>>(x.t).statement;
    auto name = std::get<parser::Name>(stmt.t);
    subPrograms_[name.ToString()] = &x;
    currentSubprogram_ = &x;
    return true;
  }

  bool Pre(parser::InterfaceBody &x) {
    insideOfInterfaceBody_ = true;
    if (auto *func{parser::Unwrap<parser::InterfaceBody::Function>(x.u)}) {
      const auto &stmt{
          std::get<parser::Statement<parser::FunctionStmt>>(func->t)};
      auto name = std::get<parser::Name>(stmt.statement.t);
      interfaces_[name.ToString()] = &x;
    }

    if (auto *subr{parser::Unwrap<parser::InterfaceBody::Subroutine>(x.u)}) {
      const auto &stmt{
          std::get<parser::Statement<parser::SubroutineStmt>>(subr->t)};
      auto name = std::get<parser::Name>(stmt.statement.t);
      interfaces_[name.ToString()] = &x;
    }

    return true;
  }
};

bool FinalizeOMP(SemanticsContext &context, parser::Program &program) {
  ImplicitDeclareTargetCapture impCap{context};
  Walk(program, impCap);
  return !context.AnyFatalError();
}

} // namespace Fortran::semantics
